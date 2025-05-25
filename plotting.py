# Import libraries
import pandas as pd
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, dcc, html, Input, Output, State, dash_table
from dash.exceptions import PreventUpdate

# Load and preprocess the Titanic dataset
df = sns.load_dataset('titanic')
df = df[['survived', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']]
df['age'] = df.groupby(['pclass', 'sex'])['age'].transform(lambda x: x.fillna(x.median()))
df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])
df['survived_label'] = df['survived'].map({0: 'Did Not Survive', 1: 'Survived'})

# Initialize the Dash app
app = Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])

# Define the layout
app.layout = html.Div([
    # Title
    html.H1("Advanced Titanic Survival Dashboard", style={'textAlign': 'center', 'color': '#2c3e50', 'fontSize': 36}),
    
    # Description
    html.P(
        "Explore survival patterns from the Titanic dataset with advanced interactive visualizations. "
        "Use filters to customize the data and interact with plots to uncover insights.",
        style={'textAlign': 'center', 'color': '#34495e', 'fontSize': 18, 'marginBottom': 20}
    ),
    
    # Filters
    html.Div([
        html.Label("Select Passenger Class:", style={'fontSize': 16, 'marginRight': 10}),
        dcc.Dropdown(
            id='pclass-filter',
            options=[
                {'label': 'All Classes', 'value': 'all'},
                {'label': '1st Class', 'value': 1},
                {'label': '2nd Class', 'value': 2},
                {'label': '3rd Class', 'value': 3}
            ],
            value='all',
            className='dcc_control'
        ),
        
        html.Label("Select Sex:", style={'fontSize': 16, 'marginRight': 10}),
        dcc.Dropdown(
            id='sex-filter',
            options=[
                {'label': 'All', 'value': 'all'},
                {'label': 'Male', 'value': 'male'},
                {'label': 'Female', 'value': 'female'}
            ],
            value='all',
            className='dcc_control'
        ),
        
        html.Label("Select Age Range:", style={'fontSize': 16, 'marginRight': 10}),
        html.Div(
            dcc.RangeSlider(
                id='age-filter',
                min=0,
                max=80,
                step=1,
                value=[0, 80],
                marks={i: str(i) for i in range(0, 81, 10)},
                tooltip={"placement": "bottom", "always_visible": True},
                allowCross=False,
                updatemode='mouseup'
            ),
            className='dcc_control'
        ),
        
        html.Label("Select Plot Type for Survival Analysis:", style={'fontSize': 16, 'marginRight': 10}),
        dcc.RadioItems(
            id='plot-type',
            options=[
                {'label': 'Bar Plot', 'value': 'bar'},
                {'label': 'Pie Chart', 'value': 'pie'}
            ],
            value='bar',
            style={'marginBottom': 20}
        ),
        
        html.Button('Reset Filters', id='reset-button', n_clicks=0, style={'fontSize': 16, 'marginTop': 10})
    ], style={'padding': 20, 'backgroundColor': '#f8f9fa'}),
    
    # Visualizations
    html.Div([
        # Combined Subplot (Survival and Age Distribution)
        html.H3("Survival and Age Analysis", style={'textAlign': 'center'}),
        dcc.Graph(id='combined-plot'),
        
        # Fare vs. Age Scatter Plot with Cross-Filtering
        html.H3("Fare vs. Age by Passenger Class and Survival", style={'textAlign': 'center'}),
        dcc.Graph(id='fare-age-scatter'),
        
        # Data Table for Selected Points
        html.H3("Selected Passenger Data", style={'textAlign': 'center'}),
        dash_table.DataTable(id='selected-data-table', page_size=5)
    ], style={'padding': 20}),
    
    # Insights
    html.Div([
        html.H3("Key Insights", style={'textAlign': 'center'}),
        html.Ul([
            html.Li("Females had a significantly higher survival rate than males, especially in 1st and 2nd class."),
            html.Li("1st-class passengers had the highest survival rates, likely due to better access to lifeboats."),
            html.Li("Younger passengers in 3rd class had lower survival rates, reflecting socioeconomic disparities.")
        ], style={'fontSize': 16, 'color': '#34495e', 'textAlign': 'left', 'margin': '0 auto', 'width': '80%'})
    ], style={'padding': 20, 'backgroundColor': '#f8f9fa'})
], style={'fontFamily': 'Arial, sans-serif', 'maxWidth': '1200px', 'margin': '0 auto'})

# Callback to reset filters
@app.callback(
    [Output('pclass-filter', 'value'),
     Output('sex-filter', 'value'),
     Output('age-filter', 'value'),
     Output('plot-type', 'value')],
    [Input('reset-button', 'n_clicks')]
)
def reset_filters(n_clicks):
    if n_clicks > 0:
        return 'all', 'all', [0, 80], 'bar'
    raise PreventUpdate

# Callback to update visualizations and table
@app.callback(
    [Output('combined-plot', 'figure'),
     Output('fare-age-scatter', 'figure'),
     Output('selected-data-table', 'data')],
    [Input('pclass-filter', 'value'),
     Input('sex-filter', 'value'),
     Input('age-filter', 'value'),
     Input('plot-type', 'value'),
     Input('fare-age-scatter', 'selectedData')]
)
def update_dashboard(pclass, sex, age_range, plot_type, selected_data):
    # Filter the dataset
    filtered_df = df.copy()
    if pclass != 'all':
        filtered_df = filtered_df[filtered_df['pclass'] == pclass]
    if sex != 'all':
        filtered_df = filtered_df[filtered_df['sex'] == sex]
    filtered_df = filtered_df[(filtered_df['age'] >= age_range[0]) & (filtered_df['age'] <= age_range[1])]
    
    # Combined Subplot (Survival and Age)
    fig_combined = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Survival Rate by Class and Sex", "Age Distribution by Class and Survival"),
        specs=[[{"type": "xy"}, {"type": "xy"}]]
    )
    
    # Survival Plot (Bar or Pie)
    survival_rates = filtered_df.groupby(['pclass', 'sex'])['survived'].mean().reset_index()
    if plot_type == 'bar':
        for sex_val in survival_rates['sex'].unique():
            sex_data = survival_rates[survival_rates['sex'] == sex_val]
            fig_combined.add_trace(
                go.Bar(
                    x=sex_data['pclass'],
                    y=sex_data['survived'],
                    name=sex_val.capitalize(),
                    hovertemplate=f'<b>Class:</b> %{{x}}<br><b>Survival Rate:</b> %{{y:.2%}}<br><b>Sex:</b> {sex_val}<extra></extra>',
                    marker=dict(opacity=0.8)
                ),
                row=1, col=1
            )
        fig_combined.update_xaxes(title_text="Passenger Class", row=1, col=1)
        fig_combined.update_yaxes(title_text="Survival Rate", tickformat=".0%", row=1, col=1)
    else:
        survival_counts = filtered_df.groupby(['pclass', 'sex', 'survived_label'])['survived'].count().reset_index(name='count')
        for pclass_val in survival_counts['pclass'].unique():
            pclass_data = survival_counts[survival_counts['pclass'] == pclass_val]
            fig_combined.add_trace(
                go.Pie(
                    labels=pclass_data['survived_label'],
                    values=pclass_data['count'],
                    name=f'Class {pclass_val}',
                    hovertemplate='<b>Survival:</b> %{label}<br><b>Count:</b> %{value}<br><b>Class:</b> '+str(pclass_val)+'<extra></extra>',
                    domain={'x': [0.25 * (pclass_val-1), 0.25 * pclass_val], 'y': [0.2, 0.8]},
                    visible=True if pclass_val == 1 else False
                ),
                row=1, col=1
            )
        # Add animation for pie chart transitions
        fig_combined.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    direction="right",
                    x=0.5,
                    y=1.2,
                    showactive=True,
                    buttons=[
                        dict(label=f"Class {i}", method="update", args=[{"visible": [True if j+1 == i else False for j in range(3)]}])
                        for i in range(1, 4)
                    ]
                )
            ]
        )
    
    # Age Distribution (Violin)
    for survived_val in filtered_df['survived_label'].unique():
        survived_data = filtered_df[filtered_df['survived_label'] == survived_val]
        fig_combined.add_trace(
            go.Violin(
                x=survived_data['pclass'],
                y=survived_data['age'],
                name=survived_val,
                box_visible=True,
                points='all',
                pointpos=-1.8,
                jitter=0.2,
                hovertemplate='<b>Class:</b> %{x}<br><b>Age:</b> %{y}<br><b>Survival:</b> '+survived_val+'<extra></extra>',
                marker=dict(size=4)
            ),
            row=1, col=2
        )
    fig_combined.update_xaxes(title_text="Passenger Class", row=1, col=2)
    fig_combined.update_yaxes(title_text="Age", row=1, col=2)
    
    # Add annotation to highlight key insight
    fig_combined.add_annotation(
        xref="paper", yref="paper",
        x=0.5, y=1.05,
        text="Click bars or points to filter the table below",
        showarrow=False,
        font=dict(size=12, color="#2c3e50")
    )
    
    fig_combined.update_layout(
        title="Survival and Age Analysis",
        title_x=0.5,
        template='plotly_white',
        showlegend=True,
        font=dict(size=12),
        height=500
    )
    
    # Fare vs. Age Scatter Plot
    fig_scatter = px.scatter(
        filtered_df,
        x='age',
        y='fare',
        color='survived_label',
        facet_col='pclass',
        title='Fare vs. Age by Passenger Class and Survival',
        labels={'age': 'Age', 'fare': 'Fare', 'survived_label': 'Survival'},
        hover_data=['sex', 'embarked'],
        template='plotly_white'
    )
    # Add shape to highlight high-fare survivors
    fig_scatter.add_shape(
        type="rect",
        x0=0, x1=80, y0=200, y1=filtered_df['fare'].max(),
        fillcolor="rgba(0, 255, 0, 0.2)",
        line=dict(width=0),
        layer="below",
        row="all", col="all"
    )
    fig_scatter.add_annotation(
        x=40, y=250,
        text="High-Fare Survivors",
        showarrow=True,
        arrowhead=2,
        ax=20, ay=-30,
        font=dict(size=10, color="#2c3e50")
    )
    fig_scatter.update_layout(title_x=0.5, font=dict(size=12), height=500)
    
    # Selected Data Table (from scatter plot cross-filtering)
    selected_points = filtered_df if selected_data is None else filtered_df.iloc[
        [point['pointIndex'] for point in selected_data['points']] if selected_data and 'points' in selected_data else []
    ]
    table_data = selected_points[['pclass', 'sex', 'age', 'fare', 'survived_label', 'embarked']].to_dict('records')
    
    return fig_combined, fig_scatter, table_data

# Run the app
if __name__ == '__main__':
    app.run(debug=True)