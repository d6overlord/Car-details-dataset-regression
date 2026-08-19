# Car-details-dataset-regression
regression
Cars — предсказание цены продажи подержанного автомобиля

Проект по очистке данных о подержанных автомобилях и построению модели линейной регрессии для предсказания цены продажи (selling_price) на основе характеристик автомобиля.

Описание

Исходный датасет cars.csv содержит объявления о продаже подержанных автомобилей: полное название модели, год выпуска, пробег, тип топлива, тип продавца, коробку передач и число предыдущих владельцев. Ноутбук cars.ipynb упрощает признак с названием автомобиля, строит Pipeline из scikit-learn с ColumnTransformer и обучает LinearRegression, а затем сохраняет модель в cars.joblib.

Файлы проекта
Файл	Описание
cars.csv	Исходный датасет, 4340 строк × 8 столбцов
cars.ipynb	Очистка данных, обучение и сохранение модели линейной регрессии
cars.joblib	Обученная модель (Pipeline с ColumnTransformer + LinearRegression), сериализованная через joblib
Исходные столбцы датасета
Столбец	Описание
name	Полное название модели (например, «Maruti 800 AC»)
year	Год выпуска
selling_price	Цена продажи (целевая переменная)
km_driven	Пробег в километрах
fuel	Тип топлива: Diesel, Petrol, CNG, LPG, Electric
seller_type	Тип продавца: Individual, Dealer, Trustmark Dealer
transmission	Коробка передач: Manual, Automatic
owner	Число предыдущих владельцев: First Owner, Second Owner, Third Owner, Fourth & Above Owner, Test Drive Car
Что делает ноутбук
1. Загрузка и первичный осмотр

Чтение CSV через pandas, просмотр типов данных и пропусков (df.info()) — в исходных данных пропусков нет.

2. Упрощение признака name

Полное название модели (например, "Maruti Wagon R LXI Minor") сведено к марке автомобиля — первому слову в строке ("Maruti"). Написана функция return_car_mark, применяющая values.split(' ')[0] к каждому значению столбца name, с проверкой на пропуски (pd.isnull).

3. Подготовка данных к обучению
Признаки (x) и целевая переменная selling_price (y) разделены на обучающую и тестовую выборки (train_test_split, test_size=0.2).
Признаки автоматически разделены на числовые (select_dtypes(['float64', 'int64']): year, km_driven) и категориальные (select_dtypes(['object']): name, fuel, seller_type, transmission, owner).
4. Pipeline и обучение модели

Построен единый Pipeline через make_pipeline:

ColumnTransformer, применяющий OneHotEncoder к категориальным признакам (name, fuel, seller_type, transmission, owner) и StandardScaler к числовым признакам (year, km_driven);
на выходе — LinearRegression.

Модель обучена через model.fit(x_train, y_train).

Результаты
MSE (Mean Squared Error): ≈ 90.7 млрд
RMSE (√MSE): ≈ 301 182 (в тех же единицах, что и selling_price, то есть средняя ошибка предсказания — около 301 тыс. валютных единиц, что довольно много относительно типичных цен в датасете)
Сохранение и инференс модели

Модель сохранена в cars.joblib через joblib.dump(). Пример инференса на новом автомобиле:

python
car_row = pd.DataFrame({
    "name": ["Toyota"],
    "year": [2019],
    "km_driven": [32000],
    "fuel": ["Diesel"],
    "seller_type": ["Dealer"],
    "transmission": ["Manual"],
    "owner": ["Second Owner"]
})
predictions = loaded_model.predict(car_row)

Результат для этого примера: ≈ 1 164 394.

Технологии
Python 3
pandas, numpy — обработка данных
scikit-learn — train_test_split, StandardScaler, OneHotEncoder, ColumnTransformer, Pipeline/make_pipeline, LinearRegression, mean_squared_error
joblib — сериализация обученной модели
Как запустить
bash
pip install pandas numpy scikit-learn joblib
jupyter notebook cars.ipynb

Убедитесь, что cars.csv находится в той же папке, что и ноутбук — путь к файлу указан относительным (pd.read_csv('cars.csv')).

Возможные улучшения
Довольно высокий RMSE (≈301 тыс.) говорит о том, что линейная регрессия плохо описывает зависимость цены от признаков — стоит попробовать более гибкие модели (RandomForestRegressor, GradientBoostingRegressor, XGBoost).
Зафиксировать random_state в train_test_split для воспроизводимости результатов (сейчас не задан).
Марка name содержит редкие категории (уникальных марок может быть много, некоторые представлены единичными объявлениями) — стоит объединить редкие марки в категорию "Other", чтобы избежать переобучения на разреженных one-hot признаках.
Учесть сильный дисбаланс классов в fuel (Electric — всего 1 объявление, LPG/CNG — единицы десятков) и в owner (Test Drive Car — 17 объявлений) — такие редкие категории плохо обучаются и могут вести себя непредсказуемо на новых данных.
Рассмотреть логарифмирование целевой переменной selling_price — цены на автомобили обычно имеют скошенное распределение (много дешёвых машин и немного очень дорогих), что может улучшить качество линейной модели.
Добавить кросс-валидацию (cross_val_score) для более надёжной оценки качества модели, а не полагаться на одно случайное разбиение.
Использовать более информативный признак вместо года выпуска — например, «возраст автомобиля» (current_year - year), который может лучше обобщаться на будущие данные.
