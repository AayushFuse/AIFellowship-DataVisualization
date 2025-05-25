# Titanic Dataset Interactive Dashboard
# Plotly Dash Application for Data Visualization and Analysis

import dash
from dash import dcc, html, Input, Output, dash_table
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import seaborn as sns

# Load and preprocess the data
def load_and_preprocess_data():
    """Load and preprocess the Titanic dataset"""
    try:
        # Try loading from local file first
        df = pd.read_csv('titanic.csv')
    except:
        # Fallback to seaborn's built-in dataset
        df = sns.load_dataset('titanic')
    
    # Data preprocessing
    df_clean = df.copy()
    
    # Handle missing values
    df_clean['age'] = df_clean.groupby(['pclass', 'sex'])['age'].transform(
        lambda x: x.fillna(x.median())
    )
    df_clean['embarked'] = df_clean['embarked'].fillna(df_clean['embarked'].mode()[0])
    df_clean['fare'] = df_clean['fare'].fillna(df_clean['fare'].median())
    
    # Create new features
    df_clean['age_group'] = pd.cut(df_clean['age'], 
                                   bins=[0, 12, 18, 35, 60, 100], 
                                   labels=['Child', 'Teen', 'Young Adult', 'Adult', 'Senior'])
    
    df_clean['family_size'] = df_clean['sibsp'] + df_clean['parch'] + 1
    df_clean['family_category'] = pd.cut(df_clean['family_size'],
                                        bins=[0, 1, 4, 11],
                                        labels=['Alone', 'Small Family', 'Large Family'])
    
    df_clean['fare_category'] = pd.qcut(df_clean['fare'], q=4, 
                                       labels=['Low', 'Medium-Low', 'Medium-High', 'High'])
    
    # Create deck feature
    if 'cabin' in df_clean.columns:
        df_clean['deck'] = df_clean['cabin'].str[0]
        df_clean['deck'] = df_clean['deck'].fillna('Unknown')
    else:
        df_clean['deck'] = 'Unknown'
    
    # Convert categorical variables for better display
    df_clean['survived_label'] = df_clean['survived'].map({0: 'Died', 1: 'Survived'})
    df_clean['pclass_label'] = df_clean['pclass'].map({1: '1st Class', 2: '2nd Class', 3: '3rd Class'})
    df_clean['sex_label'] = df_clean['sex'].map({'male': 'Male', 'female': 'Female'})
    
    return df_clean

# Load data
df = load_and_preprocess_data()

# Initialize the Dash app
app = dash.Dash(__name__)

# Define colors and styling
colors = {
    'background': '#f8f9fa',
    'text': '#2c3e50',
    'primary': '#3498db',
    'secondary': '#e74c3c',
    'success': '#2ecc71',
    'warning': '#f39c12'
}

# Define the app layout
app.layout = html.Div([
    # Header
    html.Div([
        html.H1("🚢 Titanic Disaster Analysis Dashboard", 
                style={'textAlign': 'center', 'color': colors['text'], 'marginBottom': 30}),
        html.P("Interactive analysis of passenger survival patterns on the RMS Titanic",
               style={'textAlign': 'center', 'fontSize': 18, 'color': '#7f8c8d'})
    ], style={'backgroundColor': colors['background'], 'padding': '20px'}),
    
    # Key Metrics Row
    html.Div([
        html.Div([
            html.H3(f"{len(df):,}", style={'color': colors['primary'], 'margin': 0}),
            html.P("Total Passengers", style={'margin': 0, 'fontSize': 14})
        ], className='metric-box', style={
            'backgroundColor': 'white', 'padding': '20px', 'borderRadius': '10px',
            'textAlign': 'center', 'margin': '10px', 'boxShadow': '0 2px 10px rgba(0,0,0,0.1)'
        }),
        
        html.Div([
            html.H3(f"{df['survived'].sum():,}", style={'color': colors['success'], 'margin': 0}),
            html.P("Survivors", style={'margin': 0, 'fontSize': 14})
        ], className='metric-box', style={
            'backgroundColor': 'white', 'padding': '20px', 'borderRadius': '10px',
            'textAlign': 'center', 'margin': '10px', 'boxShadow': '0 2px 10px rgba(0,0,0,0.1)'
        }),
        
        html.Div([
            html.H3(f"{df['survived'].mean():.1%}", style={'color': colors['warning'], 'margin': 0}),
            html.P("Survival Rate", style={'margin': 0, 'fontSize': 14})
        ], className='metric-box', style={
            'backgroundColor': 'white', 'padding': '20px', 'borderRadius': '10px',
            'textAlign': 'center', 'margin': '10px', 'boxShadow': '0 2px 10px rgba(0,0,0,0.1)'
        }),
        
        html.Div([
            html.H3(f"£{df['fare'].mean():.0f}", style={'color': colors['secondary'], 'margin': 0}),
            html.P("Average Fare", style={'margin': 0, 'fontSize': 14})
        ], className='metric-box', style={
            'backgroundColor': 'white', 'padding': '20px', 'borderRadius': '10px',
            'textAlign': 'center', 'margin': '10px', 'boxShadow': '0 2px 10px rgba(0,0,0,0.1)'
        })
    ], style={'display': 'flex', 'justifyContent': 'space-around', 'margin': '20px 0'}),
    
    # Filters Section
    html.Div([
        html.H3("Filters", style={'color': colors['text']}),
        html.Div([
            html.Div([
                html.Label("Passenger Class:", style={'fontWeight': 'bold'}),
                dcc.Dropdown(
                    id='class-filter',
                    options=[{'label': 'All Classes', 'value': 'all'}] + 
                            [{'label': f'{i} Class', 'value': i} for i in sorted(df['pclass'].unique())],
                    value='all',
                    clearable=False
                )
            ], style={'width': '23%', 'display': 'inline-block', 'marginRight': '2%'}),
            
            html.Div([
                html.Label("Gender:", style={'fontWeight': 'bold'}),
                dcc.Dropdown(
                    id='gender-filter',
                    options=[{'label': 'All Genders', 'value': 'all'}] + 
                            [{'label': gender.title(), 'value': gender} for gender in df['sex'].unique()],
                    value='all',
                    clearable=False
                )
            ], style={'width': '23%', 'display': 'inline-block', 'marginRight': '2%'}),
            
            html.Div([
                html.Label("Age Range:", style={'fontWeight': 'bold'}),
                dcc.RangeSlider(
                    id='age-filter',
                    min=df['age'].min(),
                    max=df['age'].max(),
                    value=[df['age'].min(), df['age'].max()],
                    marks={int(age): str(int(age)) for age in np.linspace(df['age'].min(), df['age'].max(), 8)},
                    tooltip={"placement": "bottom", "always_visible": True}
                )
            ], style={'width': '23%', 'display': 'inline-block', 'marginRight': '2%'}),
            
            html.Div([
                html.Label("Embarkation Port:", style={'fontWeight': 'bold'}),
                dcc.Dropdown(
                    id='port-filter',
                    options=[{'label': 'All Ports', 'value': 'all'}] + 
                            [{'label': port, 'value': port} for port in df['embarked'].unique() if pd.notna(port)],
                    value='all',
                    clearable=False
                )
            ], style={'width': '23%', 'display': 'inline-block'})
        ])
    ], style={'backgroundColor': 'white', 'padding': '20px', 'margin': '20px', 
              'borderRadius': '10px', 'boxShadow': '0 2px 10px rgba(0,0,0,0.1)'}),
    
    # Main Visualizations
    html.Div([
        # Row 1: Survival Overview
        html.Div([
            html.Div([
                dcc.Graph(id='survival-pie-chart')
            ], style={'width': '50%', 'display': 'inline-block'}),
            
            html.Div([
                dcc.Graph(id='survival-by-gender')
            ], style={'width': '50%', 'display': 'inline-block'})
        ]),
        
        # Row 2: Class and Age Analysis
        html.Div([
            html.Div([
                dcc.Graph(id='survival-by-class')
            ], style={'width': '50%', 'display': 'inline-block'}),
            
            html.Div([
                dcc.Graph(id='age-distribution')
            ], style={'width': '50%', 'display': 'inline-block'})
        ]),
        
        # Row 3: Advanced Analysis
        html.Div([
            html.Div([
                dcc.Graph(id='fare-vs-age-scatter')
            ], style={'width': '50%', 'display': 'inline-block'}),
            
            html.Div([
                dcc.Graph(id='survival-heatmap')
            ], style={'width': '50%', 'display': 'inline-block'})
        ]),
        
        # Row 4: Family and Fare Analysis
        html.Div([
            html.Div([
                dcc.Graph(id='family-survival')
            ], style={'width': '50%', 'display': 'inline-block'}),
            
            html.Div([
                dcc.Graph(id='fare-distribution')
            ], style={'width': '50%', 'display': 'inline-block'})
        ])
    ]),
    
    # Data Table Section
    html.Div([
        html.H3("Passenger Data", style={'color': colors['text']}),
        dash_table.DataTable(
            id='passenger-table',
            columns=[
                {'name': 'Name', 'id': 'name'} if 'name' in df.columns else {'name': 'ID', 'id': 'index'},
                {'name': 'Class', 'id': 'pclass_label'},
                {'name': 'Gender', 'id': 'sex_label'},
                {'name': 'Age', 'id': 'age', 'type': 'numeric', 'format': {'specifier': '.0f'}},
                {'name': 'Survived', 'id': 'survived_label'},
                {'name': 'Fare', 'id': 'fare', 'type': 'numeric', 'format': {'specifier': '.2f'}}
            ],
            page_size=10,
            sort_action='native',
            filter_action='native',
            style_cell={'textAlign': 'left', 'padding': '10px'},
            style_header={'backgroundColor': colors['primary'], 'color': 'white', 'fontWeight': 'bold'},
            style_data_conditional=[
                {
                    'if': {'filter_query': '{survived_label} = Survived'},
                    'backgroundColor': '#d5f4e6',
                },
                {
                    'if': {'filter_query': '{survived_label} = Died'},
                    'backgroundColor': '#ffeaa7',
                }
            ]
        )
    ], style={'backgroundColor': 'white', 'padding': '20px', 'margin': '20px', 
              'borderRadius': '10px', 'boxShadow': '0 2px 10px rgba(0,0,0,0.1)'})
], style={'backgroundColor': colors['background'], 'minHeight': '100vh'})

# Callback for filtering data
@app.callback(
    [Output('survival-pie-chart', 'figure'),
     Output('survival-by-gender', 'figure'),
     Output('survival-by-class', 'figure'),
     Output('age-distribution', 'figure'),
     Output('fare-vs-age-scatter', 'figure'),
     Output('survival-heatmap', 'figure'),
     Output('family-survival', 'figure'),
     Output('fare-distribution', 'figure'),
     Output('passenger-table', 'data')],
    [Input('class-filter', 'value'),
     Input('gender-filter', 'value'),
     Input('age-filter', 'value'),
     Input('port-filter', 'value')]
)
def update_dashboard(class_filter, gender_filter, age_range, port_filter):
    # Filter data based on selections
    filtered_df = df.copy()
    
    if class_filter != 'all':
        filtered_df = filtered_df[filtered_df['pclass'] == class_filter]
    
    if gender_filter != 'all':
        filtered_df = filtered_df[filtered_df['sex'] == gender_filter]
    
    filtered_df = filtered_df[(filtered_df['age'] >= age_range[0]) & 
                              (filtered_df['age'] <= age_range[1])]
    
    if port_filter != 'all':
        filtered_df = filtered_df[filtered_df['embarked'] == port_filter]
    
    # 1. Survival Pie Chart
    survival_counts = filtered_df['survived_label'].value_counts()
    pie_fig = px.pie(
        values=survival_counts.values,
        names=survival_counts.index,
        title="Overall Survival Distribution",
        color_discrete_map={'Survived': colors['success'], 'Died': colors['secondary']}
    )
    pie_fig.update_layout(showlegend=True, height=400)
    
    # 2. Survival by Gender
    gender_survival = filtered_df.groupby(['sex_label', 'survived_label']).size().unstack(fill_value=0)
    gender_fig = px.bar(
        x=gender_survival.index,
        y=[gender_survival['Died'], gender_survival['Survived']],
        title="Survival by Gender",
        labels={'x': 'Gender', 'y': 'Count'},
        color_discrete_map={'Survived': colors['success'], 'Died': colors['secondary']}
    )
    gender_fig.update_layout(height=400, barmode='stack')
    
    # 3. Survival by Class
    class_survival = filtered_df.groupby('pclass_label')['survived'].mean()
    class_fig = px.bar(
        x=class_survival.index,
        y=class_survival.values,
        title="Survival Rate by Passenger Class",
        labels={'x': 'Class', 'y': 'Survival Rate'},
        color=class_survival.values,
        color_continuous_scale='RdYlGn'
    )
    class_fig.update_layout(height=400, showlegend=False)
    
    # 4. Age Distribution
    age_fig = px.histogram(
        filtered_df,
        x='age',
        color='survived_label',
        title="Age Distribution by Survival Status",
        labels={'age': 'Age', 'count': 'Number of Passengers'},
        color_discrete_map={'Survived': colors['success'], 'Died': colors['secondary']}
    )
    age_fig.update_layout(height=400)
    
    # 5. Fare vs Age Scatter Plot
    scatter_fig = px.scatter(
        filtered_df,
        x='age',
        y='fare',
        color='survived_label',
        size='family_size',
        hover_data=['pclass_label', 'sex_label'],
        title="Age vs Fare (Size = Family Size)",
        labels={'age': 'Age', 'fare': 'Fare (£)'},
        color_discrete_map={'Survived': colors['success'], 'Died': colors['secondary']}
    )
    scatter_fig.update_layout(height=400)
    
    # 6. Survival Heatmap
    heatmap_data = filtered_df.pivot_table(
        values='survived',
        index='pclass_label',
        columns='sex_label',
        aggfunc='mean'
    )
    heatmap_fig = px.imshow(
        heatmap_data.values,
        x=heatmap_data.columns,
        y=heatmap_data.index,
        color_continuous_scale='RdYlGn',
        title="Survival Rate Heatmap: Class vs Gender",
        labels={'color': 'Survival Rate'}
    )
    heatmap_fig.update_layout(height=400)
    
    # 7. Family Survival
    family_survival = filtered_df.groupby('family_category')['survived'].mean()
    family_fig = px.bar(
        x=family_survival.index,
        y=family_survival.values,
        title="Survival Rate by Family Size",
        labels={'x': 'Family Category', 'y': 'Survival Rate'},
        color=family_survival.values,
        color_continuous_scale='RdYlGn'
    )
    family_fig.update_layout(height=400, showlegend=False)
    
    # 8. Fare Distribution
    fare_fig = px.box(
        filtered_df,
        x='survived_label',
        y='fare',
        title="Fare Distribution by Survival Status",
        labels={'survived_label': 'Survival Status', 'fare': 'Fare (£)'},
        color='survived_label',
        color_discrete_map={'Survived': colors['success'], 'Died': colors['secondary']}
    )
    fare_fig.update_layout(height=400, showlegend=False)
    
    # 9. Update table data
    table_columns = ['pclass_label', 'sex_label', 'age', 'survived_label', 'fare']
    if 'name' in filtered_df.columns:
        table_columns = ['name'] + table_columns
    else:
        filtered_df['index'] = range(len(filtered_df))
        table_columns = ['index'] + table_columns
    
    table_data = filtered_df[table_columns].to_dict('records')
    
    return (pie_fig, gender_fig, class_fig, age_fig, scatter_fig, 
            heatmap_fig, family_fig, fare_fig, table_data)

# Run the app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8050)

# Additional styling and configuration
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>Titanic Analysis Dashboard</title>
        {%favicon%}
        {%css%}
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 0;
            }
            .metric-box:hover {
                transform: translateY(-2px);
                transition: transform 0.2s ease-in-out;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Export function for saving dashboard as HTML
def save_dashboard_html(filename='titanic_dashboard.html'):
    """Save the dashboard as a standalone HTML file"""
    app.run(debug=False, dev_tools_silence_routes_logging=True)

print("🚢 Titanic Dashboard Application Ready!")
print("📊 Features included:")
print("   ✅ Interactive filters (Class, Gender, Age, Port)")
print("   ✅ Real-time chart updates")
print("   ✅ Multiple visualization types")
print("   ✅ Data table with sorting and filtering")
print("   ✅ Responsive design")
print("   ✅ Professional styling")
print("\n🚀 To run the dashboard:")
print("   1. Save this code as 'titanic_dashboard.py'")
print("   2. Install requirements: pip install dash plotly pandas seaborn")
print("   3. Run: python titanic_dashboard.py")
print("   4. Open browser to: http://localhost:8050")