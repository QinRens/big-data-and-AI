# -*- coding: utf-8 -*-
import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
from plotly import graph_objs as go
import pandas as pd
colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}
function=['function_1','clean_data','function_3','function_4']

app = dash.Dash()
df = pd.read_csv(
    'C:/Sublime Text 3/dash/train1.csv')
table=df.copy()
cols = df.columns
app.layout=html.Div([
    html.H1([
        html.H1(children='Data Visualization')],
        style={'textAlign': 'center',
        'color':'white'}
        ),
     html.Div([
        dcc.Input(id='my-id',
        inputmode='text'
        )],style={'display':'inline-block','width':'30%','margin':'0px'}),
    html.Div(id='my-div',style={'color':'white'}),
    html.Div(children=' '),
    html.Div([
        dcc.Dropdown(id='xialaicaidan1',
        options=[
            {'label': 'change SalePrice data', 'value': 'clean'},
            {'label': 'function_1', 'value': '1'},
            {'label': 'function_2', 'value': '2'},
            {'label': 'function_3', 'value': '3'}
        ],
        value=['1', '2'],
        multi=True
    )],style={'display':'inline-block','width':'98%','margin':'0px'}),
    html.Div([
        html.Table([html.Tr([html.Th(col) for col in df.columns])] +
           [html.Tr([
               html.Td(df.iloc[i][col]) for col in df.columns
           ]) for i in range(min(len(df),100))]
       )],
            style={'display':'inline-block','height':'400px','width':'98%','overflow':'scroll','font-size':'20px','margin':'20px','backgroundColor':'black','textAlign': 'center','color':'white'}
            ),
    html.Div([
        html.Div([
            dcc.Dropdown(
                id='crossfilter-xaxis-column',
                options=[{'label': i, 'value': i}for i in cols],
                value='YearBuilt'
            ),
            dcc.RadioItems(
                id='crossfilter-xaxis-type',
                options=[{'label': i, 'value': i} for i in ['Linear', 'Log']],
                value='Linear',
                labelStyle={'display': 'inline-block'}
            )
        ],
        style={'width': '49%', 'display': 'inline-block'}),

        html.Div([
            dcc.Dropdown(
                id='crossfilter-yaxis-column',
                options=[{'label': i, 'value': i}for i in cols],
                value='SalePrice'
            ),
            dcc.RadioItems(
                id='crossfilter-yaxis-type',
                options=[{'label': i, 'value': i} for i in ['Linear', 'Log']],
                value='Linear',
                labelStyle={'display': 'inline-block'}
            )
        ], style={'width': '49%', 'float': 'right', 'display': 'inline-block'})
    ], style={
        'borderBottom': 'thin lightgrey solid',
        'backgroundColor': 'rgb(250, 250, 250)',
        'padding': '10 5'
    }),


    html.Div([
        dcc.Graph(
            id='graph1',
            style={
                'position':'inline-block',
                'width':'95%',
                'height':'30%',
                'margin':'10px',
                'backgroundColor':'black',
                'plot_bgcolor':colors['background'],
                'paper_bgcolor': colors['background'],
                'font':{
                    'color': colors['text']}
        }
        ),

    dcc.Graph(
        id='graph2',
        style={
            'position':'inline-block',
            'width':'98%',
            'height':'30%',
            'margin':'0px',
            'backgroundColor':'black',
            'plot_bgcolor':colors['background'],
                'paper_bgcolor': colors['background'],
                'font':{
                    'color': colors['text']}
        }
        ),
    dcc.Graph(
        id='graph3',
        style={
        'float':'right',
            'width':'49%',
            'height':'25%',
            'margin':'0px',
            'backgroundColor':'black',
                'plot_bgcolor':colors['background'],
                'paper_bgcolor': colors['background'],
                'font':{
                    'color': colors['text']}
        }
        ),
    dcc.Graph(
        id='graph4',
        style={
        'position':'inline-block',
            'width':'49%',
            'height':'25%',
            'margin':'0px',
            'backgroundColor':'black',
                'plot_bgcolor':colors['background'],
                'paper_bgcolor': colors['background'],
                'font':{
                    'color': colors['text']}
        }
        ), dcc.Graph(
        id='graph5',
        style={
        'float':'right',
            'width':'49%',
            'height':'25%',
            'margin':'0px',
            'backgroundColor':'black',
                'plot_bgcolor':colors['background'],
                'paper_bgcolor': colors['background'],
                'font':{
                    'color': colors['text']}}
        ),
        dcc.Graph(
        id='graph6',
        style={'position':'inline-block',
            'width':'49%',
            'height':'30%',
            'margin':'0px',
            'backgroundColor':'black',
                'plot_bgcolor':colors['background'],
                'paper_bgcolor': colors['background'],
                'font':{
                    'color': colors['text']}}
        ),
    
   
],style={'backgroundColor':'black'})
    ],style={'backgroundColor':'black'})


@app.callback(
    Output('my-div', 'children'),
    [Input('my-id', 'value')]
)
def update_output_div(input_value):
    return 'You choose to show "{}" rows'.format(input_value)


@app.callback(
    dash.dependencies.Output('graph1','figure'),
    [dash.dependencies.Input('xialaicaidan1', 'value'),
    dash.dependencies.Input('crossfilter-xaxis-column', 'value'),
     dash.dependencies.Input('crossfilter-yaxis-column', 'value'),
     dash.dependencies.Input('crossfilter-xaxis-type', 'value'),
     dash.dependencies.Input('crossfilter-yaxis-type', 'value')
     ]
     )    
def update_graph(value,xaxis_column_name, yaxis_column_name,
                 xaxis_type, yaxis_type):
    if 'clean' in value:
        for i in range(0,len(df)):
            df['SalePrice'][i]=10000
        print('SalePrice has been changed')
    if len(df[xaxis_column_name].unique())<=26:
        type_list=[]
        for each in df[xaxis_column_name].unique():
            type_list.append(each)
        for j in range(0,len(df)):
            for i in range(0,len(type_list)):
                if df[xaxis_column_name][j]==type_list[i]:
                    df[xaxis_column_name][j]=i
    return {
        'data': [go.Scatter(
            x=df[xaxis_column_name],
            y=df[yaxis_column_name],
            text=df[yaxis_column_name],
            customdata=df[yaxis_column_name],
            mode='markers',
            marker={
                'size': 12,
                'opacity': 0.4,
                'line': {'width': 0.5, 'color': 'white'}
            }
        )
        ],
        'layout': go.Layout(      
            xaxis={
                'title': xaxis_column_name,
                'type': 'linear' if xaxis_type == 'Linear' else 'log'
            },
            yaxis={
                'title': yaxis_column_name,
                'type': 'linear' if yaxis_type == 'Linear' else 'log'
            },
            margin={'l': 70, 'b': 70, 't': 10, 'r': 0},
            height=450,
            hovermode='closest',
            plot_bgcolor= colors['background'],
            paper_bgcolor= colors['background'],
            font={
                'color': colors['text']
        })}

@app.callback(
    dash.dependencies.Output('graph2','figure'),
    [dash.dependencies.Input('crossfilter-xaxis-column', 'value'),
     dash.dependencies.Input('crossfilter-yaxis-column', 'value')
    ])
def update_graph_graph2(xaxis_column_name, yaxis_column_name
                 ):
    dic={}
    num={}
    avg=[]
    for i in range(0,len(table)-1):
        dic[table[xaxis_column_name][i]]=0
    for i in range(0,len(table)-1):
        num[table[xaxis_column_name][i]]=0
    for i in range(0,len(table)-1):
        num[table[xaxis_column_name][i]]=1+num[table[xaxis_column_name][i]]
    for i in range(0,len(table)-1):
        dic[table[xaxis_column_name][i]]=table[yaxis_column_name][i]+dic[table[xaxis_column_name][i]]
    for key in num:
        avg.append(dic[key]/num[key])
    return{
    'data': [go.Bar(
            x=sorted(table[xaxis_column_name].unique()),
            y=avg,
            marker={
                'line': {'width': 0}},
            showlegend=True
        )],
        'layout': go.Layout(      
            xaxis={
                'title': xaxis_column_name
            },
            yaxis={
                'title': yaxis_column_name
            },
            margin={'l': 70, 'b': 70, 't': 40, 'r': 0},
            height=450,
            hovermode='closest',
            title='The avearge of each data',
             plot_bgcolor= colors['background'],
                paper_bgcolor= colors['background'],
                font={
                    'color': colors['text']
        
    })}
@app.callback(
    dash.dependencies.Output('graph3','figure'),
    [dash.dependencies.Input('crossfilter-xaxis-column','value'),
    dash.dependencies.Input('crossfilter-yaxis-column','value'),
    dash.dependencies.Input('xialaicaidan1','value')
    ])
def update_graph_graph3(xaxis_column_name, yaxis_column_name,value):
    
    labels=sorted(table[xaxis_column_name])
    counts = {}
    times=[]
    each_1=[]
    for x in labels:
        if x in counts:
            counts[x] += 1
        else:
            counts[x] = 1
    for each in counts:
        each_1.append(each)
        times.append(counts[each])
    if len(table[xaxis_column_name].unique())<=26:
        return {
        'data':[go.Pie(
            labels=each_1,
            values=times
            )],
        'layout': go.Layout(      
                xaxis={
                    'title': xaxis_column_name
                },
                yaxis={
                    'title': yaxis_column_name
                },
                title='The number of each data',
                height=450,
                hovermode='closest',
                margin={'l': 70, 'b': 70, 't': 40, 'r': 0},
                plot_bgcolor= colors['background'],
                paper_bgcolor= colors['background'],
                font={
                    'color': colors['text']
            }
        )}
    else:
        return {'data':[go.Pie(
            labels=['NAN'],
            values=[1]
            )],
        'layout': go.Layout(
                xaxis={
                    'title': xaxis_column_name
                },
                yaxis={
                    'title': yaxis_column_name
                },
                title='The number of each data',
            margin={'l': 70, 'b': 70, 't': 40, 'r': 0},
                height=450,
                hovermode='closest',
                plot_bgcolor= colors['background'],
                paper_bgcolor= colors['background'],
                font={
                    'color': colors['text']
        })}

@app.callback(
    dash.dependencies.Output('graph4','figure'),
    [dash.dependencies.Input('crossfilter-xaxis-column','value'),
    dash.dependencies.Input('crossfilter-yaxis-column','value')
    ])
def update_graph_graph4(xaxis_column_name, yaxis_column_name):
    
    if len(table[xaxis_column_name].unique())<=26:
        return {
        'data':[go.Histogram(
            x=sorted(table[xaxis_column_name]),
            marker={
                'line': {'width': 0}}
            )],
        'layout': go.Layout(       #da tu shu ju
                xaxis={
                    'title': xaxis_column_name
                },
                yaxis={
                    'title': yaxis_column_name
                },
                title='The number of each data',
                margin={'l': 40, 'b': 30, 't': 50, 'r': 20},
                height=450,
                hovermode='closest',
                  plot_bgcolor= colors['background'],
                paper_bgcolor= colors['background'],
                font={
                    'color': colors['text']
        }
            )
        }
    else:
        return{ 'data':[go.Histogram(
            x=[0],
            )],
        'layout': go.Layout(       #da tu shu ju
                xaxis={
                    'title': xaxis_column_name
                },
                yaxis={
                    'title': yaxis_column_name
                },
                title='The number of each data',
                margin={'l': 40, 'b': 30, 't': 50, 'r': 20},
                height=450,
                hovermode='closest',
                  plot_bgcolor= colors['background'],
                paper_bgcolor= colors['background'],
                font={
                    'color': colors['text']
        }
            )}
@app.callback(
    dash.dependencies.Output('graph5','figure'),
    [dash.dependencies.Input('crossfilter-xaxis-column','value'),
    dash.dependencies.Input('crossfilter-yaxis-column','value')
    ])
def update_graph_graph5(xaxis_column_name, yaxis_column_name):
    labels=sorted(table[xaxis_column_name])
    counts = {}
    times=[]
    each_1=[]
    labels_1=[]
    for x in labels:
        if x in counts:
            counts[x] += 1
        else:
            counts[x] = 1
    for each in counts:
        each_1.append(each)
        times.append(counts[each])
    if len(table[xaxis_column_name].unique())<=26:
        a=sorted(table[xaxis_column_name].unique())
        b=times
        a.append(a[0])
        b.append(b[0])
        print(a)
        print(b)
        max1=max(b)
        min1=min(b)
        return {
        'data' : [go.Scatterpolar(
              r = b,
              theta = a,
              fill = 'toself'
            )],

        'layout': go.Layout(
              polar = dict(
                radialaxis = dict(
                  visible = True,
                  range = [min1, max1]
                )
              ),
              showlegend = True,
              plot_bgcolor= colors['background'],
                paper_bgcolor= colors['background'],
                font={
                    'color': colors['text']}
            ),
        }
    else:
        return {'data':[go.Pie(
            labels=['NAN'],
            values=[1]
            )],
        'layout': go.Layout(       #da tu shu ju
                xaxis={
                    'title': xaxis_column_name
                },
                yaxis={
                    'title': yaxis_column_name
                },
                title='The number of each data',
                margin={'l': 40, 'b': 30, 't': 50, 'r': 20},
                height=450,
                hovermode='closest',
                plot_bgcolor= colors['background'],
                paper_bgcolor= colors['background'],
                font={
                    'color': colors['text']}
            )}
@app.callback(
    dash.dependencies.Output('graph6','figure'),
    [dash.dependencies.Input('crossfilter-xaxis-column','value'),
    dash.dependencies.Input('crossfilter-yaxis-column','value')
    ])
def update_graph_graph6(xaxis_column_name, yaxis_column_name):
    return {
    'data':[
        go.Histogram2dContour(
            x = table[xaxis_column_name],
            y = table[yaxis_column_name],
            colorscale = 'Blues',
            reversescale = True,
            xaxis = 'x',
            yaxis = 'y'
        ),
    ],

    'layout':go.Layout(
        autosize = False,
        xaxis = dict(
            zeroline = False,
            showgrid = False
        ),
        yaxis = dict(
            zeroline = False,
            
            showgrid = False
        ),
        xaxis2 = dict(
    zeroline = False,
  
    showgrid = False
),
yaxis2 = dict(
    zeroline = False,

    showgrid = False
),
        hovermode = 'closest',
        showlegend = True,
          plot_bgcolor= colors['background'],
            paper_bgcolor= colors['background'],
        margin={'l': 70, 'b': 70, 't': 40, 'r': 0},
            font={
                'color': colors['text']
    }
    )}
if __name__ == '__main__':
    app.run_server(debug=True)