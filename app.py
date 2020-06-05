import time
import os
import dash
import base64
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
from dash.dependencies import Input, Output
import gunicorn

import data_for_dash

_dd = data_for_dash.manupilation_data()

# External Style datasheet
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
#mathjax = 'https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/MathJax.js?config=TeX-MML-AM_CHTML'
#app.scripts.append_script({ 'external_url' : mathjax })

server = app.server

colors = {
    'background':'#E6F2F2',
    'font2':'#3C8969',
    'text': '#2F34E8'
}

image_filename = os.path.dirname(os.path.abspath(__file__))+'//LNLS_sem_fundo.png'
encoded_image = base64.b64encode(open(image_filename, 'rb').read())

app.layout = html.Div(style={'backgroundColor': colors['background']}, children=[
    html.Img(src='data:image/png;base64,{}'.format(encoded_image.decode()),
             style={'width': '20%','float': 'center', 'display': 'inline-block'}),
    html.H2(children='Sirius Undulator Synchrotron Radiation Maths', style={'width': '80%','float': 'right', 'display': 'inline-block'}),

    html.Div(children='Dashboard with practical applications for calculating machine parameters and synchrotron radiation from undulator light sources. Developed by the LNLS/SIRIUS magnets group.', style={
        'textAlign': 'left',
        'color': colors['text']
        }),
    
    dcc.Markdown('''
    > - **Energy:** 0.1 - 2.5 keV;
    > - **Type:** Delta;
    > - **Period (mm):** 52.5;
    > - **Length (m):** 2.4;
    > - **Kmax:** 5.88
    '''),

    dcc.Markdown('''
    **SIRUS Storage Ring Specification Parameters**
    '''),
    html.Div([  #Open Machine Div
        html.Div([  #First table column
            html.H6('Beamline'),
            dcc.Input(id='beamline', placeholder='Enter a value...', value = 'Sabiá', type='text'
            ),

            html.H6('Undulator type'),
            dcc.Input(id='undulator_type', placeholder='Enter a value...', value = 'Delta', type='text'
            ),
            
            html.H6('Electron Energy (GeV)'),
            dcc.Input(id='electron-energy', placeholder='Enter a value...', value = 3, type='number'
            ),

            html.H6('Average Current (A)'),
            dcc.Input(id='avg_current', placeholder='Enter a value...', value = 0.350, type='number'
            ),

            html.H6('Circumference (m)'),
            dcc.Input(id='Circum', placeholder='Enter a value...', value = 518.396, type='number'
            ),

            html.H6('Bunches'),
            dcc.Input(id='Bunches', placeholder='Enter a value...', value = 864, type='number'
            ),

            html.H6('sigma z (mm)'),
            dcc.Input(id='sigma_z', placeholder='Enter a value...', value = 11.6, type='number'
            )
            
        ],
        style={'width': '50%', 'display': 'inline-block'}),

        html.Div([  #Second table column
            html.H6('Natural Emittance (m.rad)'),
            dcc.Input(id='nat_emittance', placeholder='Enter a value...', value = 0.25192064617379e-9, type='number'
            ),

            html.H6('Coupling Constant'),
            dcc.Input(id='coupling_cte', placeholder='Enter a value...', value = 0.01, type='number'
            ),
            
            html.H6('Energy Spread'),
            dcc.Input(id='energy_spread', placeholder='Enter a value...', value = 0.00084666891478399, type='number'
            ),

            html.H6('Beta_x (m)'),
            dcc.Input(id='beta_x', placeholder='Enter a value...', value = 1.357, type='number'
            ),

            html.H6('Beta_y (m)'),
            dcc.Input(id='beta_y', placeholder='Enter a value...', value = 1.6, type='number'
            ),

            html.H6('Eta_x (m)'),
            dcc.Input(id='eta_x', placeholder='Enter a value...', value = 0, type='number'
            ),

            html.H6('Eta_y (m)'),
            dcc.Input(id='eta_y', placeholder='Enter a value...', value = 0, type='number'
            ),
        ],style={'width': '50%', 'float': 'right', 'display': 'inline-block'}),

        html.Div([
            html.H5(children='Storage Ring Parameters Calculation')   
        ],style={'textAlign': 'center','width': '100%', 'float': 'center', 'color': colors['font2']})
        
    ] + [html.Div(id="out-all-types")]), #Close Machine Div

    ### Undulator parameters
    html.Div([
            html.H5(children='Undulator Parameters')   
        ],style={'textAlign': 'center','width': '100%', 'float': 'center', 'color': colors['text']}),

    html.Div([ # Open Undulator Div
        html.Div([  
            html.H6('Gap value (mm)'),
            dcc.Input(id='gap_value', placeholder='Enter a value...', value = 13.6, type='number'
            ),

            html.H6('B (T)'),
            dcc.Input(id='magnetic_field', placeholder='Enter a value...', value = 1.2, type='number'
            ),

            html.H6('Periodic Length (mm)'),
            dcc.Input(id='periodic_length', placeholder='Enter a value...', value = 52.5, type='number'
            ),
            
            html.H6('Device Length (m)'),
            dcc.Input(id='device_length', placeholder='Enter a value...', value = 2.4, type='number'
            ),

        ], style={'textAlign': 'center','width': '100%', 'float': 'center'})

    ] + [html.Div(id="out-undulator")]), #Close Undulator Div
    

    html.Div([ # Opne Div DataFrame
        html.Div(children=[
            html.H5(children='Summary table with first 15th odd harmonics'),
            #Shows Data frame here
            ],style={'textAlign': 'center','width': '100%'})

        ]+ [html.Div(id="out-dataframe")]), #Close Div Dataframe

    dcc.Markdown('''
    **Select the Flux or Brightness as a function of photon energy graph:**
    '''),

    dcc.Dropdown(id='cb_graph', 
    options=[
        {'label': 'Brightness', 'value': 'Bright'},
        {'label': 'Flux', 'value': 'Flux'}
    ],
    value='Flux'
    ),

    html.Div([ #Open Div graph
        html.Div([
        dcc.Graph(figure=_dd.type_plot(), id='my-figure'),
        html.Label('Slider for photon energy (keV)'),
        dcc.RangeSlider(
            id='graph-range-slider',
            min=0,
            max=25.00,
            step=0.5,
            value=[0, 20]
        ),
        html.Div(id='output-container-range-slider')
        ],style={'width': '100%', 'display': 'inline-block'})
    ])#Close Div Graph

]) #Close app.layout


@app.callback(
    Output("out-all-types", "children"),
    [Input("electron-energy", "value"),
     Input("avg_current", "value"),
     Input("Circum", "value"),
     Input("Bunches", "value"),
     Input("sigma_z", "value"),
     Input("nat_emittance", "value"),
     Input("coupling_cte", "value"),
     Input("energy_spread", "value"),
     Input("beta_x", "value"),
     Input("beta_y", "value"),
     Input("eta_x", "value"),
     Input("eta_y", "value")]
)
def cb_render(*vals):
    machine_values = []
    for val in vals:
        if not isinstance(val, (int, float)):
            val = 0.001
        machine_values.append(float(val))
    return _dd.machine_calculus(machine_values[0], machine_values[1], machine_values[2],
                                machine_values[3], machine_values[4], machine_values[5],
                                machine_values[6], machine_values[7], machine_values[8],
                                machine_values[9], machine_values[10], machine_values[11])

@app.callback(
    Output("out-undulator", "children"),
    [Input("gap_value", "value"),
     Input("magnetic_field", "value"),
     Input("periodic_length", "value"),
     Input("device_length", "value")]
    )

def update_output(*vals):
    undulator_values = []
    for val in vals:
        if not isinstance(val, (int, float)):
            val = 0.001
        undulator_values.append(float(val))
    time.sleep(0.05)
    return _dd.undulator_calculus(undulator_values[0], undulator_values[1], undulator_values[2], undulator_values[3])

@app.callback(
    Output("out-dataframe","children"),
    [Input("out-undulator", "children")]
    )
def update_tab(vals):
    return _dd.dataframe_all()

#Callback for graph
@app.callback(
    Output(component_id='my-figure', component_property='figure'),
    [Input(component_id='cb_graph', component_property='value'),
     Input(component_id='graph-range-slider', component_property='value')]
    )
def update_graph(*vals):
    '''prepare figure from value'''
    graph_vals = []
    for val in vals:
        graph_vals.append(val)
    return _dd.type_plot(graph_vals[0], graph_vals[1])


if __name__ == '__main__':
    app.run_server(debug=True)
    
