import faicons as fa
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from scipy import stats
from shiny import reactive, render
from shiny.express import input, ui
from shinywidgets import render_plotly
import seaborn as sns
from sklearn import metrics
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
 

df = pd.read_csv("data/output.csv")

price_rng = (df.price.min(), df.price.max())

df['date'] = pd.to_datetime(df['date'])
date_rng = (df['date'].agg(['min', 'max']))

сities = list(df['city'].unique())
correlations = list(df.select_dtypes('number').columns.values)

with_outliers = True

def replace_outliers_with_nan_iqr(df, feature, inplace=False):
    desired_feature = df[feature]
    
    q1, q3 = desired_feature.quantile([0.25, 0.75])
    iqr = q3 - q1
    upper_bound = q3 + 1.5 * iqr
    lower_bound = q1 - 1.5 * iqr
    indices = (desired_feature[(desired_feature > upper_bound) | (desired_feature < lower_bound)]).index
    if not inplace:
        return desired_feature.replace(desired_feature[indices].values, np.nan)
    return desired_feature.replace(desired_feature[indices].values, np.nan, inplace=True)

features_with_outlier = ['price', 'sqft_lot', 'sqft_basement']


ui.page_opts(title="Прогнозирование цен (Финальный проект)", fillable=True)

with ui.sidebar(open="desktop"):
    ui.input_checkbox("outliers", "Выбросы", with_outliers)
    ui.input_slider(
        "house_price",
        "Диапазон цен",
        min=price_rng[0],
        max=price_rng[-1],
        value=(price_rng[0], price_rng[-1]),
        pre="$",
    )
    ui.input_date_range(
        "daterange", 
        "Диапазон дат", 
        start=date_rng[0],
        end=date_rng[1] - pd.Timedelta(days=1),
        min=date_rng[0],
        max=date_rng[1],
        language='ru'
    )
    ui.input_checkbox_group(
        "data",
        "Города",
        list(df['city'].unique()),
        selected=сities,
        inline=True,
    )
    ui.input_action_button("reset", "Сбросить фильтры")

ICONS = {
    "ellipsis": fa.icon_svg("ellipsis"),
}

ui.markdown(
    """
    ### Описание функционала
    - Все результаты изменяются **динамически** при изменении значения фильтров.
    - По умолчанию выбросы включены в отображение. Чтобы их отключить, нужно снять значение **Выбросы**.
    - При необходимости, фильтры можно сбросить и вернуться к начальному состоянию. Для этого, необходимо нажать **Сбросить фильтры**
    - Каждый элемент (карточку) можно раскрыть на весь экран. Для этого необходимо навести на элемент и выбрать в правом нижнем углу **Expand**.
    - У элемента **корреляции** есть дополнительные фильтры, чтобы ими воспользоваться неоходимо нажать на **...** и выбрать интересующие параметры. 
   ---
    """
)

ui.markdown(
    """
    ### Исследование данных
    ###### Выведем данные в виде таблицы, а также посмотрим на описательную статистику. 
    """
)

with ui.layout_columns(col_widths=[6, 6, 12], fill=False):
    with ui.card(full_screen=True):
        ui.card_header("Данные по домам")

        @render.data_frame
        def table():
            return render.DataGrid(data())
        
    with ui.card(full_screen=True):
        ui.card_header("Описательная статистика данных")

        @render.table
        def table2():
            return data().describe()
        
ui.markdown(
    """
    ###### Из таблиц выше, видно, что значения функций `price`, `sqft_lot` и `sqft_basement` распределены неравномерно; они с большей вероятностью будут содержать крайние значения. Следовательно, мы извлечем эти признаки и обработаем их выбросы.
    """
)
        
with ui.layout_columns(col_widths=[12, 12, 12], fill=False):  
    with ui.card(full_screen=True):    
        with ui.card_header(class_="d-flex justify-content-between align-items-center"):
            "Корреляция между числовыми признаками"
            with ui.popover(title="Добавить/убрать признак", placement="top"):
                ICONS["ellipsis"]
                ui.input_checkbox_group(
                    "correlations",
                    None,
                    list(df.select_dtypes('number').columns.values),
                    selected=correlations,
                    inline=True,
                )

        @render.plot
        def heatmap():
            x = data_corr().select_dtypes('number')
            return sns.heatmap(x.corr(), annot=True)

ui.markdown(
    """
    ###### Из корреляции выше, видно, что среди числовых признаков, самое значимое влияние на цену имеют признаки `sqft_above`, `sqft_living`, `sqft_lot`, `yr_built`, `yr_renovated`.
    """
)

with ui.layout_columns(col_widths=[12, 12, 12], fill=False):  
    with ui.card(full_screen=True):    
        with ui.card_header(class_="d-flex justify-content-between align-items-center"):
            "Распределение цен"

        @render_plotly
        def hist_hypo():
            data_df = hypothesis()
            fig = go.Figure()

            fig.add_trace(go.Histogram(x=data_df['price']))


            mean_price = data_df['price'].mean()
            median_price = data_df['price'].median()

            fig.add_vline(x=mean_price, line_dash = 'dash', line_color = 'Red')
            fig.add_vline(x=median_price, line_dash = 'dash', line_color = 'Green')


            fig.update_layout(
                xaxis_title="Price",
                yaxis_title="Density",
                legend_title="Legend"
            )

            return fig
        
        with ui.card_footer():
            "Проверим нулевую гипотезу Тест Шапиро-Уилка о том, что данные взяты из нормального распределения."

            @render.code
            def footer():
                statistic, pvalue = shapiro_test_results()
                
                return f"\n statistic={statistic}, pvalue={pvalue}"

ui.markdown(
    """
    ###### Из графика выше, можно выделить 3 характеристики.
    - Распределение отличается от нормального
    - Распределение правостороннее
    - Пик приходится на более низкую цену продажи (по сравнению с максимальной ценой продажи)
    """
)   
        

with ui.layout_columns(col_widths=[6, 6, 12], fill=False):  
    with ui.card(full_screen=True):    
        with ui.card_header(class_="d-flex justify-content-between align-items-center"):
            "Распределение видов у домов"

        @render_plotly
        def hist():
            return px.histogram(
                data(),
                x="view",
            )
        
    with ui.card(full_screen=True):    
        with ui.card_header(class_="d-flex justify-content-between align-items-center"):
            "Цена и вид"

        @render_plotly
        def box_plot():
            return px.box(
                data(),
                x="view",
                y="price",
            )
ui.markdown(
    """
    ###### Из графиков выше, видно, что у большинства домов в этом наборе данных нет вида. Цены на дома с видом выше, чем без него.
    """
)       


with ui.layout_columns(col_widths=[6, 6, 12], fill=False):
    with ui.card(full_screen=True):    
        with ui.card_header(class_="d-flex justify-content-between align-items-center"):
            "Распределение по городам"

        @render_plotly
        def hist_city():
            return px.histogram(
                data_sort(),
                x="city",
                y="count",
            )      
                 
    with ui.card(full_screen=True):    
        with ui.card_header(class_="d-flex justify-content-between align-items-center"):
            "Cредняя цена на жилье в каждом городе"

        @render_plotly
        def hist_avg_mean():
            return px.histogram(
                hist_avg(),
                x="city",
                y="price",
            )

ui.markdown(
    """
    ###### Из графиков выше, видно, что большинство домов в `Сиэтле`, а также, что диапазон цен сильно отличается.
    """
)     


ui.markdown(
    """
    ---
    ### Линейная регрессия
    ###### Построим модель линейной регрессии и спрогнозируем ценообразование на основании ключевых признаков. 
    ###### В качестве ключевых признаков были выбраны: `bedrooms`, `sqft_living`, `sqft_lot`, `sqft_basement`, `yr_built`, `yr_renovated`, `city`.
    """
)        
        
with ui.layout_columns(col_widths=[12, 12, 12], fill=False):         
    with ui.card(full_screen=True):
        with ui.card_header(class_="d-flex justify-content-between align-items-center"):
            "Фактические и прогнозируемые цены (Линейная регрессия)"

        @render_plotly
        def scatterplot():
            return px.scatter(
                data(),
                x=lineral_model()[0],
                y=lineral_model()[1],
                labels={'x': 'Фактические', 'y': 'Прогнозируемые'},
            )

with ui.layout_columns(col_widths=[6, 6, 12], fill=False):
    with ui.card(full_screen=True):
        ui.card_header("Коэффиценты признаков")

        @render.data_frame
        def table3():
            return render.DataGrid(coef())
        
    with ui.card(full_screen=True):
        ui.card_header("Метрики качества")

        @render.data_frame
        def table4():
            return render.DataGrid(calc_metrics())
        
ui.markdown(
    """
    ---
    ### Выводы
    ###### Жилая площадь, год постройки и город являются ключевыми факторами, влияющими на цену недвижимости.
    """
)        
            
            

        
@reactive.calc
def data():
    price = input.house_price()
    idx1 = df[df['price'].between(price[0], price[1])]

    date = input.daterange()
    mask = (df['date'] > pd.to_datetime(date[0])) & (df['date'] <= pd.to_datetime(date[1]))
    idx2 = df.loc[mask]

    idx3 = list(input.data())

    df_filter = pd.merge(idx1, idx2, how='inner', on=['price', 'date'], suffixes=('', '_y')) 
    df_filter.drop(df_filter.filter(regex='_y$').columns, axis=1, inplace=True)
    
    df_filter['date'] = df_filter['date'].dt.strftime('%Y-%m-%d')


    with_outliers = input.outliers()

    if not with_outliers:
        df_without_outliers = df_filter.copy()
        features_means = df_without_outliers[features_with_outlier].mean()

        for i in features_with_outlier:
            replace_outliers_with_nan_iqr(df_without_outliers, i, inplace=True)

        df_without_outliers.fillna(features_means, inplace=True)
        df_without_outliers.isnull().sum().sum()

        return df_without_outliers[df_without_outliers['city'].isin(idx3)]
    else:

        return df_filter[df_filter['city'].isin(idx3)]

@reactive.calc
def data_corr():
    data_df = data()

    correlations = input.correlations()

    return data_df[list(correlations)]

@reactive.calc
def data_sort():
    data_df = data()

    sorted_counts = data_df['city'].value_counts().sort_values()

    sorted_df = pd.DataFrame({'city': sorted_counts.index, 'count': sorted_counts.values})

    return sorted_df

@reactive.calc
def hist_avg():
    data_df = data()

    df_group = data_df.groupby('city')['price'].mean().reset_index().sort_values(by='price')


    return df_group

@reactive.calc
def hypothesis():
    data_df = data()

    return data_df


@reactive.calc
def coef():
    df_coef = lineral_model()[2]

    df_coef['Признак'] = df_coef.index

    return df_coef

@reactive.calc
def shapiro_test_results():
    data_df = hypothesis()
    statistic, pvalue = stats.shapiro(data_df['price'])
    return statistic, pvalue

@reactive.calc
def calc_metrics():
    y_pred, y_true = lineral_model()[0], lineral_model()[1]
    data = [
        ['MAE', metrics.mean_absolute_error(y_true, y_pred)],
        ['MSE', metrics.mean_squared_error(y_true, y_pred)],
        ['MAPE', metrics.mean_absolute_percentage_error(y_true, y_pred)],
        ['MAD', stats.median_abs_deviation(y_pred)],
        ['R2', metrics.r2_score(y_true, y_pred)]
    ]

    columns = ['Метрика', 'Значение']

    df = pd.DataFrame(data, columns=columns, dtype=object)

    return df

@reactive.calc
def lineral_model():
    data_df = data().copy()

    data_df.drop(columns=['statezip','country',"date","street"],inplace=True)
    data_df.drop(columns=["sqft_above","bathrooms"],inplace=True)
    data_df.drop(columns=['floors',"waterfront",'condition'],inplace=True)
    data_df.drop(columns=["view"],inplace=True)

    label_encoder = LabelEncoder()
    data_df['city'] = label_encoder.fit_transform(data_df['city'])

    X=data_df.drop(columns=['price'])
    y=data_df['price']

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

    pipeline = make_pipeline(StandardScaler(),LinearRegression())
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    lr_model = pipeline.named_steps['linearregression']

    coefficients = lr_model.coef_
    intercept = lr_model.intercept_

    df_coef = pd.DataFrame(data=coefficients, index=X.columns, columns=["Коэффицент"])


    return y_pred, y_test, df_coef, y_train

        
@reactive.effect
@reactive.event(input.reset)
def _():
    ui.update_slider("house_price", value=price_rng)
    ui.update_date_range("daterange", start=date_rng[0], end=date_rng[1] - pd.Timedelta(days=1))
    ui.update_checkbox_group("data", selected=сities)
    ui.update_checkbox_group("correlations", selected=correlations)
    ui.update_checkbox("outliers", value=with_outliers)