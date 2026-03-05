import pandas as pd
import re
import numpy as np
from sentence_transformers import SentenceTransformer, util
import torch
import os
from tqdm import tqdm  # для прогресс-бара (нужно установить: pip install tqdm)


def create_technical_errors_file_large(input_file, output_file, similarity_threshold=0.7, chunk_size=10000):
    """
    Оптимизированная версия для больших файлов
    Обрабатывает файл по частям (chunks)
    """

    # Загружаем модель для семантического поиска
    print("Загрузка NLP модели...")
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    # Регулярное выражение
    pattern = re.compile(
        r'техническ[а-я]+\s+ошибк[а-я]+|ошибк[а-я]+\s+техническ[а-я]+',
        re.IGNORECASE | re.UNICODE
    )

    # Референсные фразы (кэшируем их эмбеддинги)
    reference_phrases = [
        "техническая ошибка", "ошибка в системе", "сбой программы",
        "проблема с сервером", "баг в приложении", "некорректная работа сайта",
        "ошибка в работе сервиса", "технический сбой"
    ]
    reference_embeddings = model.encode(reference_phrases, convert_to_tensor=True)

    try:
        # Проверяем существование файла
        if not os.path.exists(input_file):
            print(f"Ошибка: Файл '{input_file}' не найден.")
            return

        # Определяем формат файла по расширению
        file_ext = os.path.splitext(input_file)[1].lower()

        # Для больших файлов используем построчное чтение
        print(f"Чтение файла: {input_file}")

        # Получаем общее количество строк для прогресс-бара
        if file_ext == '.csv':
            total_rows = sum(1 for _ in open(input_file, 'r', encoding='utf-8')) - 1  # минус заголовок
        else:
            # Для Excel сначала прочитаем только чтобы узнать размер
            temp_df = pd.read_excel(input_file, nrows=1)
            # Примерная оценка (не точная, но для прогресс-бара сойдет)
            total_rows = 100000  # запасное значение

        # Список для хранения результатов
        result_chunks = []
        total_processed = 0
        regex_count = 0
        semantic_count = 0

        # Обрабатываем файл чанками
        if file_ext == '.csv':
            # Для CSV используем чанки
            chunks = pd.read_csv(input_file, chunksize=chunk_size, encoding='utf-8')
        else:
            # Для Excel читаем весь файл, но обрабатываем чанками
            df_full = pd.read_excel(input_file)
            chunks = [df_full[i:i + chunk_size] for i in range(0, len(df_full), chunk_size)]

        print("Начало обработки...")

        # Обрабатываем каждый чанк
        for chunk_idx, chunk in enumerate(tqdm(chunks, desc="Обработка чанков")):
            # Проверяем наличие необходимых столбцов
            required_columns = ['текст_ответа', 'группа']
            missing_cols = [col for col in required_columns if col not in chunk.columns]
            if missing_cols:
                print(f"\nОшибка: В файле отсутствуют столбцы: {missing_cols}")
                print(f"Доступные столбцы: {', '.join(chunk.columns)}")
                return

            # Обрабатываем текущий чанк
            text_responses = chunk['текст_ответа'].fillna('').astype(str)

            # Regex фильтр
            regex_mask = text_responses.apply(lambda x: bool(pattern.search(x)))

            # Семантический поиск (только для непустых текстов)
            non_empty_mask = text_responses.str.len() > 0
            semantic_mask = np.zeros(len(chunk), dtype=bool)
            semantic_scores = np.zeros(len(chunk))

            if non_empty_mask.any():
                texts_to_analyze = text_responses[non_empty_mask].tolist()

                # Кодируем батчами для экономии памяти
                batch_size = 32
                all_scores = []

                for i in range(0, len(texts_to_analyze), batch_size):
                    batch_texts = texts_to_analyze[i:i + batch_size]
                    batch_embeddings = model.encode(batch_texts, convert_to_tensor=True)

                    for emb in batch_embeddings:
                        similarities = util.pytorch_cos_sim(emb, reference_embeddings)
                        max_similarity = torch.max(similarities).item()
                        all_scores.append(max_similarity)

                # Заполняем маски
                semantic_scores[non_empty_mask] = all_scores
                semantic_mask[non_empty_mask] = np.array(all_scores) >= similarity_threshold

            # Комбинируем
            combined_mask = regex_mask | semantic_mask

            # Обновляем счетчики
            regex_count += regex_mask.sum()
            semantic_count += (semantic_mask & ~regex_mask).sum()

            # Если есть совпадения, обрабатываем чанк
            if combined_mask.any():
                filtered_chunk = chunk[combined_mask].copy()

                # Добавляем метаданные
                filtered_chunk['тип_обнаружения'] = 'неизвестно'
                chunk_regex_mask = regex_mask[combined_mask]
                filtered_chunk.loc[chunk_regex_mask, 'тип_обнаружения'] = 'regex'

                semantic_only = (semantic_mask[combined_mask] & ~regex_mask[combined_mask])
                filtered_chunk.loc[semantic_only, 'тип_обнаружения'] = 'нейросеть'

                filtered_chunk['уверенность_нейросети'] = 0.0
                semantic_indices = np.where(semantic_only)[0]
                for idx in semantic_indices:
                    filtered_chunk.iloc[idx, filtered_chunk.columns.get_loc('уверенность_нейросети')] = \
                        semantic_scores[combined_mask][idx]

                # Меняем группу
                filtered_chunk['группа'] = 'техническая ошибка'

                # Сохраняем чанк
                result_chunks.append(filtered_chunk)

            total_processed += len(chunk)
            if total_processed % (chunk_size * 5) == 0:
                print(f"Обработано {total_processed} строк...")

        # Объединяем все результаты
        if result_chunks:
            final_df = pd.concat(result_chunks, ignore_index=True)

            # Сохраняем результат
            print(f"\nСохранение результатов в {output_file}...")

            # Для больших файлов используем оптимизированное сохранение
            if len(final_df) > 100000:
                # Если результатов много, сохраняем в CSV (быстрее)
                csv_output = output_file.replace('.xlsx', '.csv')
                final_df.to_csv(csv_output, index=False, encoding='utf-8-sig')
                print(f"Сохранено в CSV (для больших данных): {csv_output}")
            else:
                final_df.to_excel(output_file, index=False, engine='openpyxl')
                print(f"Сохранено в Excel: {output_file}")

            # Статистика
            print(f"\n{'=' * 50}")
            print(f"РЕЗУЛЬТАТЫ ОБРАБОТКИ")
            print(f"{'=' * 50}")
            print(f"Всего обработано строк: {total_processed}")
            print(f"Найдено строк с техническими ошибками: {len(final_df)}")
            print(f"  - Найдено regex: {regex_count}")
            print(f"  - Найдено нейросетью: {semantic_count}")
            print(f"{'=' * 50}")

            # Примеры
            if len(final_df) > 0:
                print("\nПримеры найденных записей:")
                examples = final_df[['id_заявки', 'текст_ответа', 'тип_обнаружения']].head(5)
                for _, row in examples.iterrows():
                    text = str(row['текст_ответа'])[:100] + "..." if len(str(row['текст_ответа'])) > 100 else str(
                        row['текст_ответа'])
                    print(f"ID: {row['id_заявки']} | Тип: {row['тип_обнаружения']}")
                    print(f"Текст: {text}\n")
        else:
            print("Не найдено ни одной записи с техническими ошибками.")

    except Exception as e:
        print(f"Произошла ошибка: {str(e)}")
        import traceback
        traceback.print_exc()


def main_large():
    """Запуск оптимизированной версии"""
    input_file = input('Введите имя исходного файла: ')

    # Автоматически определяем имя выходного файла
    if input_file.endswith('.xlsx'):
        output_file = input_file.replace('.xlsx', '_technical_errors.xlsx')
    elif input_file.endswith('.csv'):
        output_file = input_file.replace('.csv', '_technical_errors.csv')
    else:
        output_file = "technical_errors_result.xlsx"

    # Настройки для больших файлов
    similarity_threshold = 0.65  # Порог сходства
    chunk_size = 5000  # Размер чанка (можно увеличить/уменьшить в зависимости от памяти)

    create_technical_errors_file_large(input_file, output_file, similarity_threshold, chunk_size)


if __name__ == "__main__":
    # Установите необходимые пакеты:
    # pip install tqdm openpyxl

    main_large()
