import pandas as pd
from math import sqrt
from collections import Counter

data = pd.read_csv(
    "Предпочтение пользователей Iphone или android.csv"
)

print("Количество записей в датасете:", len(data))


time_column = data.columns[0]

target_column = data.columns[-1]

budget_column = data.columns[8]

multi_column = data.columns[4]

activity_column = data.columns[15]

for column in data.select_dtypes(include="object").columns:
    data[column] = data[column].astype(str).str.strip()

def normalize_multi(value):
    parts = [
        part.strip()
        for part in str(value).split(";")
        if part.strip()
    ]

    parts.sort()

    return ";".join(parts)


data[multi_column] = data[multi_column].apply(normalize_multi)

data[activity_column] = (
    data[activity_column]
    .str.lower()
    .str.strip()
)

data.loc[
    data[budget_column] < 1000,
    budget_column
] *= 1000



X_original = data.drop(
    columns=[time_column, target_column]
).copy()

y = data[target_column].reset_index(drop=True)

numeric_columns = X_original.select_dtypes(
    include="number"
).columns.tolist()

categorical_columns = [
    column
    for column in X_original.columns
    if column not in numeric_columns
]

minimums = X_original[numeric_columns].min()
maximums = X_original[numeric_columns].max()


X = pd.get_dummies(
    X_original,
    columns=categorical_columns,
    dtype=float
)

training_columns = X.columns.tolist()

for column in numeric_columns:

    minimum = minimums[column]
    maximum = maximums[column]

    if maximum != minimum:

        X[column] = (
            X[column] - minimum
        ) / (
            maximum - minimum
        )

    else:
        X[column] = 0


X = X.reset_index(drop=True)

def distance(a, b):

    return sqrt(
        sum(
            (x - y) ** 2
            for x, y in zip(a, b)
        )
    )


def knn_test_predict(X, y, test_index, k):

    test_object = X.iloc[test_index].tolist()

    distances = []

    for i in range(len(X)):

        if i == test_index:
            continue

        train_object = X.iloc[i].tolist()

        dist = distance(
            test_object,
            train_object
        )

        distances.append(
            (dist, y.iloc[i])
        )

    distances.sort(
        key=lambda item: item[0]
    )

    nearest = distances[:k]

    votes = Counter(
        phone
        for dist, phone in nearest
    )

    return votes.most_common(1)[0][0]


print()
print("Проверяем разные значения k:")

k_values = [1, 3, 5, 7, 9]

results = {}


for k in k_values:

    correct = 0

    for i in range(len(X)):

        prediction = knn_test_predict(
            X,
            y,
            i,
            k
        )

        if prediction == y.iloc[i]:
            correct += 1

    accuracy = correct / len(X)

    results[k] = accuracy

    print(
        f"k = {k}: "
        f"точность = {accuracy * 100:.2f}%"
    )


best_k = max(
    results,
    key=results.get
)


print()
print("Лучшее значение k:", best_k)

print(
    "Точность:",
    round(
        results[best_k] * 100,
        2
    ),
    "%"
)


def choose_option(column, question):

    options = sorted(
        data[column]
        .dropna()
        .unique()
        .tolist()
    )

    print()
    print(question)

    for number, option in enumerate(options, start=1):
        print(f"{number} - {option}")

    while True:

        try:
            answer = int(
                input("Ваш выбор: ")
            )

            if 1 <= answer <= len(options):
                return options[answer - 1]

            print("Введите номер одного из вариантов.")

        except ValueError:
            print("Нужно ввести число.")


def read_number(question, minimum=None, maximum=None):

    while True:

        try:

            text = input(
                "\n" + question + "\nВаш ответ: "
            )

            value = float(
                text.replace(",", ".")
            )

            if minimum is not None and value < minimum:
                print(
                    f"Значение должно быть не меньше {minimum}."
                )
                continue

            if maximum is not None and value > maximum:
                print(
                    f"Значение должно быть не больше {maximum}."
                )
                continue

            return value

        except ValueError:
            print("Введите число.")


def read_multi_answer():

    options = set()

    for value in data[multi_column]:

        for part in value.split(";"):
            options.add(part.strip())

    options = sorted(options)

    print()
    print(
        "4. За что вы готовы переплатить, "
        "покупая смартфон?"
    )

    print(
        "Можно выбрать несколько вариантов."
    )

    for number, option in enumerate(options, start=1):
        print(f"{number} - {option}")

    while True:

        answer = input(
            "Введите номера через запятую, например 1,3: "
        )

        try:

            numbers = [
                int(number.strip())
                for number in answer.split(",")
            ]

            if not all(
                1 <= number <= len(options)
                for number in numbers
            ):
                print("Один из номеров указан неправильно.")
                continue

            selected = [
                options[number - 1]
                for number in numbers
            ]

            selected.sort()

            return ";".join(selected)

        except ValueError:
            print(
                "Введите номера вариантов через запятую."
            )


print()

print(
    "\nОтветьте на вопросы, "
    "и программа попробует определить, "
    "что вы выберете: Айфон или Андроид."
)


answers = {}


# Вопрос 1
answers[data.columns[1]] = choose_option(
    data.columns[1],
    "1. Стабильность или новшество?"
)


# Вопрос 2
answers[data.columns[2]] = choose_option(
    data.columns[2],
    "2. Часто ли вы фотографируете?"
)


# Вопрос 3
answers[data.columns[3]] = choose_option(
    data.columns[3],
    "3. Ведете ли вы блог?"
)


# Вопрос 4
answers[data.columns[4]] = read_multi_answer()


# Вопрос 5
answers[data.columns[5]] = choose_option(
    data.columns[5],
    "5. Статус или технологичность?"
)


# Вопрос 6
answers[data.columns[6]] = choose_option(
    data.columns[6],
    '6. VPN или "запрет"?'
)


# Вопрос 7
answers[data.columns[7]] = read_number(
    "7. Сколько вам лет?",
    minimum=1
)


# Вопрос 8
budget = read_number(
    "8. Сколько в среднем вы готовы "
    "потратить на новый телефон?\n"
    "Можно написать, например, 80000 или 80 "
    "(80 тысяч).",
    minimum=0
)

if budget < 1000:
    budget *= 1000

answers[data.columns[8]] = budget


# Вопрос 9
answers[data.columns[9]] = read_number(
    "9. Насколько вам важна возможность "
    "кастомизации интерфейса? От 1 до 10.",
    minimum=1,
    maximum=10
)


# Вопрос 10
answers[data.columns[10]] = choose_option(
    data.columns[10],
    "10. Можете ли вы поставить рингтон из файла?"
)


# Вопрос 11
answers[data.columns[11]] = choose_option(
    data.columns[11],
    "11. Как вы относитесь к перепродаже "
    "своего смартфона спустя 2-3 года?"
)


# Вопрос 12
answers[data.columns[12]] = choose_option(
    data.columns[12],
    "12. Насколько вам важна возможность "
    "настройки системы под себя?"
)


# Вопрос 13
answers[data.columns[13]] = read_number(
    "13. Как часто вы меняете телефон? "
    "Введите количество лет.",
    minimum=0
)


# Вопрос 14
answers[data.columns[14]] = choose_option(
    data.columns[14],
    "14. Насколько важна актуальность "
    "и обслуживание старых версий ОС?"
)


# Вопрос 15
print()
activity = input(
    "15. Ваша сфера деятельности?\n"
    "Например: IT, студент, продажи\n"
    "Ваш ответ: "
)

answers[data.columns[15]] = (
    activity.strip().lower()
)


# Вопрос 16
answers[data.columns[16]] = choose_option(
    data.columns[16],
    "16. Какого вы пола?"
)


# Вопрос 17
answers[data.columns[17]] = choose_option(
    data.columns[17],
    "17. Вы ведете Instagram?"
)


# Вопрос 18
answers[data.columns[18]] = read_number(
    "18. Сколько раз в день вы примерно "
    "заряжаете телефон?",
    minimum=0
)


# Вопрос 19
answers[data.columns[19]] = choose_option(
    data.columns[19],
    "19. Что бы вы выбрали: экономить "
    "и откладывать или порадовать себя?"
)


# Вопрос 20
answers[data.columns[20]] = choose_option(
    data.columns[20],
    "20. Ваша гарнитура от одного бренда "
    "или от разных?"
)


# Вопрос 21
answers[data.columns[21]] = read_number(
    "21. Как часто вы кушаете яблоки? "
    "Оцените от 1 до 10.",
    minimum=1,
    maximum=10
)


user_data = pd.DataFrame(
    [answers]
)

user_encoded = pd.get_dummies(
    user_data,
    columns=categorical_columns,
    dtype=float
)

user_encoded = user_encoded.reindex(
    columns=training_columns,
    fill_value=0
)

for column in numeric_columns:

    minimum = minimums[column]
    maximum = maximums[column]

    if maximum != minimum:

        user_encoded[column] = (
            user_encoded[column] - minimum
        ) / (
            maximum - minimum
        )

    else:
        user_encoded[column] = 0


def predict_new_user(
        X,
        y,
        new_user,
        k
):

    new_object = new_user.iloc[0].tolist()

    distances = []

    for i in range(len(X)):

        train_object = X.iloc[i].tolist()

        dist = distance(
            new_object,
            train_object
        )

        distances.append(
            (dist, y.iloc[i])
        )

    distances.sort(
        key=lambda item: item[0]
    )

    nearest = distances[:k]

    votes = Counter(
        phone
        for dist, phone in nearest
    )

    prediction = votes.most_common(1)[0][0]

    return prediction, votes


prediction, votes = predict_new_user(
    X,
    y,
    user_encoded,
    best_k
)


print()

print(
    f"\nИспользовано ближайших соседей: {best_k}"
)

print("\nГолоса ближайших пользователей:")

for phone, count in votes.items():
    print(
        f"{phone}: {count}"
    )


print()
print(
    "ПРОГНОЗ ПРОГРАММЫ:",
    prediction.upper()
)

print()

if prediction == "Айфон":

    print(
        "По вашим ответам вы больше похожи "
        "на пользователей iPhone."
    )

else:

    print(
        "По вашим ответам вы больше похожи "
        "на пользователей Android."
    )