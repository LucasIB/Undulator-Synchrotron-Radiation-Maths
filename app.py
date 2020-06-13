import time
import os
import dash
import base64
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
from dash.dependencies import Input, Output
import gunicorn

from reportfiles import ReportCreator as _rr
import data_for_dash

_dd = data_for_dash.manupilation_data()

#_rr = reportfiles.ReportCreator()

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

straight_section = {
    'L_Beta' : [1.357, 1.6],
    'H_beta' : [17.779, 3.566]
}

low_beta_phaseI = {
    'current_A': 0.1,
    'sigma_z': 2.5,
    'emitt': 0.25023415605855e-9,
    'coupl': 0.01,
    'espread': 0.00083    
    }

low_beta_phaseII = {
    'current_A': 0.35,
    'sigma_z': 11.6,
    'emitt': 0.15e-9,
    'coupl': 0.01,
    'espread': 0.00084358942070141   
    }


image_filename = os.path.dirname(os.path.abspath(__file__))+'//images//LNLS_sem_fundo.png'
encoded_image = base64.b64encode(open(image_filename, 'rb').read())

app.layout = html.Div(style={'backgroundColor': colors['background']}, children=[
    html.Img(src='data:image/png;base64,{}'.format(encoded_image.decode()),
             style={'width': '20%','float': 'center', 'display': 'inline-block'}),
    html.H2(children='Sirius Undulator Synchrotron Radiation Maths', style={'width': '80%','float': 'right', 'display': 'inline-block'}),

    html.Div(children='Dashboard with practical applications for calculating machine parameters and synchrotron radiation from undulator light sources. Developed by the LNLS/SIRIUS magnets group.', style={
        'textAlign': 'left',
        'color': colors['text']
        }),
    
##    dcc.Markdown('''
##    > - **Energy:** 0.1 - 2.5 keV;
##    > - **Type:** Delta;
##    > - **Period (mm):** 52.5;
##    > - **Length (m):** 2.4;
##    > - **Kmax:** 5.88
##    '''),

    html.Div([ #Open general Div combo boxes
        html.Div([
            html.H6('Sirius Phase'),
            dcc.Dropdown(id='cb_phases', 
                options=[
                    {'label': 'Phase-I', 'value': 'phase1'},
                    {'label': 'Phase-II', 'value': 'phase2'}
                ],
                value='phase1'
                )
            ],style={'width': '50%', 'display': 'inline-block'}),

        html.Div([
            html.H6('Straight Section'),
            dcc.Dropdown(id='cb_straight', 
                options=[
                    {'label': 'High-beta', 'value': 'H_beta'},
                    {'label': 'Low-beta', 'value': 'L_Beta'}
                ],
                value='L_Beta'
                )            
            ],style={'width': '50%', 'float': 'right', 'display': 'inline-block'})

        ] + [html.Div(id="out-phase-section")]),#Close general Div combo boxes

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

            html.H6('Bunch length (mm)'),
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
            html.H5(children='Undulator Input Parameters')   
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
    

    html.Div([ # Open Div DataFrame
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
            html.Label('Y-Axis type'),
            dcc.RadioItems(
                id='y-axis-scale',
                options=[
                    {'label': 'Linear', 'value': 'linear'},
                    {'label': 'Logarithm', 'value':'log'}
                    ],
                value = 'log',
                labelStyle={'display': 'inline-block'}
                ),
            html.Label('Slider for photon energy (keV)', style={'textAlign': 'center','width': '100%', 'float': 'center'}),
            dcc.RangeSlider(
                id='graph-range-slider',
                min=0,
                max=25.00,
                step=0.5,
                value=[0, 20]
            ),
            
            dcc.Graph(figure=_dd.type_plot(), id='my-figure')
            
        ],style={'width': '100%', 'display': 'inline-block'})
    ]),#Close Div Graph

    html.H5('Beam Stay Clear Straight Section', style={'textAlign': 'center','width': '100%', 'float': 'center'}),

    html.Div([ #Open Div BSC
        html.Div([
            dcc.Graph(figure=_dd.plot_BSC(), id='bsc-figure')
            ])

        ]),
    
    html.H6('Save results in PDF', style={'textAlign': 'center','width': '100%', 'float': 'center'}),
    html.Div([ # Open general div for button to export
        html.Div([
            html.Button('Create Report', id='report-call', n_clicks=0, value='True')
            ], style={'textAlign': 'center','width': '100%', 'float': 'center'})
            #_dd.report_mode(value)
        
        ] + [html.Div(id='container-button-timestamp')])

]) #Close app.layout

@app.callback(
    [Output("beta_x","value"),
     Output("beta_y","value")],
    [Input("cb_straight","value")]
    )
def refresh_tabs(value):
    values = straight_section[value]
    return values[0], values[1]

@app.callback(
    [Output("avg_current", "value"),
     Output("sigma_z", "value"),
     Output("nat_emittance", "value"),
     Output("coupling_cte", "value"),
     Output("energy_spread", "value")],
    [Input("cb_phases", "value")]
    )
def refesh_tabs2(value):
    if value == 'phase1':
       _values =  list(low_beta_phaseI.values())
    elif value == 'phase2':
        _values = list(low_beta_phaseII.values())
    return _values[0], _values[1], _values[2], _values[3], _values[4] 


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
    time.sleep(0.05)
    return _dd.dataframe_all()

#Callback save image
@app.callback(
    Output(component_id='container-button-timestamp', component_property='children'),
    [Input(component_id='report-call', component_property='n_clicks')]
    )
def refresh_save(n_clicks):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'report-call' in changed_id:
        msg = 'Report file saved'
        time.sleep(0.1)
        # Create pdf file
        if n_clicks > 0:
            _dd.report_mode(_rr)
    return msg


#Callback for graph
@app.callback(
    Output(component_id='my-figure', component_property='figure'),
    [Input(component_id='cb_graph', component_property='value'),
     Input(component_id='graph-range-slider', component_property='value'),
     Input(component_id='y-axis-scale', component_property='value')]
    )
def update_graph(*vals):
    '''prepare figure from value'''
    graph_vals = []
    for val in vals:
        graph_vals.append(val)
    return _dd.type_plot(graph_vals[0], graph_vals[1], graph_vals[2])


if __name__ == '__main__':
    app.run_server(debug=True)
    
