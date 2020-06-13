import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import dash_core_components as dcc
import dash_html_components as html
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from plotly.subplots import make_subplots
from scipy.special import jv
import reportfiles

##_rr = reportfiles.ReportCreator()

class manupilation_data(object):
    def __init__(self):
        self.variables()

    def variables(self):
        '''Establishing the variables'''
        self.electron_energy = 0
        self.avg_current = 0
        self.Circum = 0
        self.Bunches = 0
        self.sigma_z = 0
        self.nat_emittance = 0
        self.coupling_cte = 0
        self.energy_spread = 0
        self.beta_x = 0
        self.beta_y = 0
        self.eta_x = 0
        self.eta_y = 0
        self.gap_value = 1
        self.magnetic_field = 1
        self.periodic_length = 1
        self.device_lenght = 1
        self.relative_energy = 1
        self.deflection_parameter_K = 1
        self.source_size_x = 1
        self.source_size_y = 1
        self.source_divergence_x = 1
        self.source_divergence_y = 1
        self.electron_energy = 1
        self.avg_current = 1
        self.effec_size_x = 1
        self.effec_size_y = 1
        self.effec_diver_x = 1
        self.effec_diver_y = 1

    def machine_calculus(self, electron_energy, avg_current, Circum, Bunches, sigma_z,
                      nat_emittance, coupling_cte, energy_spread, beta_x, beta_y,
                      eta_x, eta_y):
        '''Main parameters calculus'''

        # machine variables
        self.electron_energy = electron_energy
        self.avg_current = avg_current
        self.Circum = Circum
        self.Bunches = Bunches
        self.sigma_z = sigma_z
        self.nat_emittance = nat_emittance
        self.coupling_cte = coupling_cte
        self.energy_spread = energy_spread
        self.beta_x = beta_x
        self.beta_y = beta_y
        self.eta_x = eta_x
        self.eta_y = eta_y

        # Relative energy
        self.relative_energy = self.electron_energy*1e9 / 0.511e06

        #Emission angle (rad)
        self.emission_angle = 1 / self.relative_energy

        # Emittance horizontal x (m.rad)
        self.emittance_horizontal_x = self.nat_emittance

        # Emittance vertical y (m.rad)
        self.emittance_vertical_y = self.nat_emittance*self.coupling_cte

        # Source Size x (m)
        self.source_size_x = np.sqrt((self.beta_x*self.nat_emittance)+(self.energy_spread*self.eta_x)**2)

        # Source Size y (m)
        self.source_size_y = np.sqrt(self.beta_y*self.emittance_vertical_y)

        # Source divergence x
        self.source_divergence_x = np.sqrt(self.nat_emittance/self.beta_x)

        # Source divergence y
        self.source_divergence_y = np.sqrt(self.emittance_vertical_y/self.beta_y)


        return html.Div([
            html.Table([
                html.Tr([html.Td(html.H6('Relative particle energy')), html.Td(np.format_float_scientific(np.float32(self.relative_energy)))]),
                html.Tr([html.Td(html.H6('Emission angle (rad)')), html.Td(np.format_float_scientific(np.float32(self.emission_angle)))]),
                html.Tr([html.Td(html.H6('Emittance horizontal x (m.rad)')), html.Td(np.format_float_scientific(np.float32(self.emittance_horizontal_x)))]),
                html.Tr([html.Td(html.H6('Emittance vertical y (m.rad)')), html.Td(np.format_float_scientific(np.float32(self.emittance_vertical_y)))]),
                html.Tr([html.Td(html.H6('Source Size x (m)')), html.Td(np.format_float_scientific(np.float32(self.source_size_x)))]),
                html.Tr([html.Td(html.H6('Source Size y (m)')), html.Td(np.format_float_scientific(np.float32(self.source_size_y)))]),
                html.Tr([html.Td(html.H6('Source divergence x (rad)')), html.Td(np.format_float_scientific(np.float32(self.source_divergence_x)))]),
                html.Tr([html.Td(html.H6('Source divergence y (rad)')), html.Td(np.format_float_scientific(np.float32(self.source_divergence_y)))]),
                
                ], style={'textAlign': 'center', 'border': '2px solid grey','width': '100%', 'float': 'center'})

        ],style={'textAlign': 'center', 'width': '100%', 'float': 'center'})

    
    def undulator_calculus(self, gap_value, magnetic_field, periodic_length, device_lenght):
        '''Calculus based in horizontal polarization'''

        # Undulator variables
        self.gap_value = gap_value
        self.magnetic_field = magnetic_field
        self.periodic_length = periodic_length
        self.device_lenght = device_lenght


        # Deflection parameter
        self.deflection_parameter_K = 0.0934*self.magnetic_field*self.periodic_length

        # Wavelenght 1º harmonic (theta=0) [m]
        self.wavelength_first_harm = (self.periodic_length/(2*1*self.relative_energy**2))*(1+(self.deflection_parameter_K**2/2)+self.relative_energy**2*(0))*(1e-3)

        # Photon energy of 1º harmonic [keV]
        self.first_photon_energy = (9.5*1*self.electron_energy**2)/(self.periodic_length*(1+(self.deflection_parameter_K**2/2)+self.relative_energy**2*(0)))

        # Photon energy of 15º harmonic [keV]
        self.fifteen_photon_energy = (9.5*15*self.electron_energy**2)/(self.periodic_length*(1+(self.deflection_parameter_K**2/2)+self.relative_energy**2*(0)))

        # Total Power radiated by an undulator [kW]
        self.power_undulator = 0.633*self.magnetic_field**2*self.electron_energy**2*self.avg_current*self.device_lenght

        # Angular ditribution of the radiation power - power/solid angle [W/mrad²]
        F_n_k = lambda k, n:((k**2*n**2)/(1 + k**2/2)**2)*(jv(((n-1)/2), ((n*k**2)/(4+2*k**2))) - jv(((n+1)/2), (n*k**2)/(4+2*k**2)))**2 #Bessel Special Function 1º order

        QnK = lambda FnK, k, n:((1 + (k**2/2))/n)*FnK #Bessel Special Function 2º order
        
        self.angular_power = 10.84*self.magnetic_field*self.electron_energy**4*self.avg_current*self.device_lenght*(self.device_lenght/(self.periodic_length*1e-3))*QnK(F_n_k(float(self.deflection_parameter_K),1), float(self.deflection_parameter_K), 1)

        # Number of photons per solid angle (Angular spectral Flux - ph/s/mrad²/0.1%BW)
        self.undulator_spectral_flux = 1.744e14*(self.device_lenght/(self.periodic_length*1e-3))**2*self.electron_energy**2*self.avg_current*F_n_k(float(self.deflection_parameter_K),1)

        # Photon Flux Undulator first harmonic [photon/s/0.1%]
        self.first_photon_flux = 1.43e14*(self.device_lenght/(self.periodic_length*1e-3))*self.avg_current*QnK(F_n_k(float(self.deflection_parameter_K),1), float(self.deflection_parameter_K), 1)

        # Undulator source size for fundamental harmonic n=1 [m]   
        self.undu_size_n1 = np.sqrt(self.wavelength_first_harm*self.device_lenght)/(4*np.pi)

        # Undulator source divergence for fundamental harmonic n=1 [m]
        self.undu_divergence_n1 = np.sqrt(self.wavelength_first_harm/self.device_lenght)

        # Effective photon beam sizes x, n=1 [m]
        self.effec_size_x = np.sqrt(self.source_size_x**2 + self.undu_size_n1**2)

        # Effective photon beam sizes y, n=1 [m]
        self.effec_size_y = np.sqrt(self.source_size_y**2 + self.undu_size_n1**2)

        # Effective photon beam divergences x, n=1 [m]
        self.effec_diver_x = np.sqrt(self.source_divergence_x**2 + self.undu_divergence_n1**2)

        # Effective photon beam divergences y, n=1 [m]
        self.effec_diver_y = np.sqrt(self.source_divergence_y**2 + self.undu_divergence_n1**2)

        # Undulator Brightness 1st, hence: [ph/s/mrad²/mm²/0.1%BW]
        self.bright_undulator_n1 = self.first_photon_flux/(4*np.pi**2*self.effec_size_x*1e3*self.effec_size_y*1e3*self.effec_diver_x*1e3*self.effec_diver_y*1e3)
             
        return html.Div([ # Open Div undulator calculus
            html.Table([ # Left Table
                html.Tr([html.Td(dcc.Markdown('''**Deflection parameter K**''')), html.Td(np.format_float_scientific(np.float32(self.deflection_parameter_K)))]),
                html.Tr([html.Td(dcc.Markdown('''**Wavelenght 1º harmonic (theta=0) [m]**''')), html.Td(np.format_float_scientific(np.float32(self.wavelength_first_harm)))]),
                html.Tr([html.Td(dcc.Markdown('''**Photon energy of 1º harmonic [keV]**''')), html.Td(np.format_float_scientific(np.float32(self.first_photon_energy)))]),
                html.Tr([html.Td(dcc.Markdown('''**Total Power radiated [kW]**''')), html.Td(np.format_float_scientific(np.float32(self.power_undulator)))]),
                html.Tr([html.Td(dcc.Markdown('''**Angular ditribution radiation power [W/mrad²]**''')), html.Td(np.format_float_scientific(np.float32(self.angular_power)))]),
                html.Tr([html.Td(dcc.Markdown('''**Angular spectral Flux [ph/s/mrad²/0.1%BW]**''')), html.Td(np.format_float_scientific(np.float32(self.undulator_spectral_flux)))]),
                html.Tr([html.Td(dcc.Markdown('''**Photon Flux Undulator 1st [photon/s/0.1%]**''')), html.Td(np.format_float_scientific(np.float32(self.first_photon_flux)))]),
                
                ], style={'width': '50%', 'border': '2px solid grey', 'display': 'inline-block'}),

            html.Table([ # Right Table
                html.Tr([html.Td(dcc.Markdown('''**Undulator source size n=1 [m]**''')), html.Td(np.format_float_scientific(np.float32(self.undu_size_n1)))]),
                html.Tr([html.Td(dcc.Markdown('''**Undulator source divergence n=1 [m]**''')), html.Td(np.format_float_scientific(np.float32(self.undu_divergence_n1)))]),
                html.Tr([html.Td(dcc.Markdown('''**Photon beam sizes x, n=1 [m]**''')), html.Td(np.format_float_scientific(np.float32(self.effec_size_x)))]),
                html.Tr([html.Td(dcc.Markdown('''**Photon beam sizes y, n=1 [m]**''')), html.Td(np.format_float_scientific(np.float32(self.effec_size_y)))]),
                html.Tr([html.Td(dcc.Markdown('''**Photon beam divergences x, n=1 [m]**''')), html.Td(np.format_float_scientific(np.float32(self.effec_diver_x)))]),
                html.Tr([html.Td(dcc.Markdown('''**Photon beam divergences y, n=1 [m]**''')), html.Td(np.format_float_scientific(np.float32(self.effec_diver_y)))]),
                html.Tr([html.Td(dcc.Markdown('''**Brightness 1st [ph/s/mrad²/mm²/0.1%BW]**''')), html.Td(np.format_float_scientific(np.float32(self.bright_undulator_n1)))]),

                ], style={'width': '50%', 'border': '2px solid grey', 'float': 'right', 'display': 'inline-block'})

        ],style={'textAlign': 'center', 'float': 'center'})

    def dataframe_all(self):
        '''Pandas Dataframe with main radiation values'''
  
        _n_harmonics = np.arange(1,16,1)
        _n_wavelengths, _n_energys = np.array([]), np.array([])
        _n_flux, _n_brigth = np.array([]), np.array([])

        F_n_k = lambda k, n:((k**2*n**2)/(1 + k**2/2)**2)*(jv(((n-1)/2), ((n*k**2)/(4+2*k**2))) - jv(((n+1)/2), (n*k**2)/(4+2*k**2)))**2 #Bessel Special Function

        QnK = lambda FnK, k, n:((1 + (k**2/2))/n)*FnK #Bessel Special Function

        for i in range(len(_n_harmonics)):
            #Wavelength of nth harmonic
            _n_wavelengths = np.append(_n_wavelengths, (self.periodic_length/(2*_n_harmonics[i]*self.relative_energy**2))*(1+(self.deflection_parameter_K**2/2)+self.relative_energy**2*(0))*(1e-3))
            
            #Photon sizes
            _undu_size_X = np.sqrt(self.source_size_x**2 + (np.sqrt(_n_wavelengths[i]*self.device_lenght)/(4*np.pi))**2)
            _undu_size_Y = np.sqrt(self.source_size_y**2 + (np.sqrt(_n_wavelengths[i]*self.device_lenght)/(4*np.pi))**2)
            
            #Photon divergence
            _undu_diver_X = np.sqrt(self.source_divergence_x**2 + (np.sqrt(_n_wavelengths[i]/self.device_lenght))**2)
            _undu_diver_Y = np.sqrt(self.source_divergence_y**2 + (np.sqrt(_n_wavelengths[i]/self.device_lenght))**2)
            
            #Photon energy of nth harmonic
            _n_energys = np.append(_n_energys, (9.5*_n_harmonics[i]*self.electron_energy**2)/(self.periodic_length*(1+(self.deflection_parameter_K**2/2)+self.relative_energy**2*(0))))
            
            #Flux
            _n_flux = np.append(_n_flux, 1.43e14*(self.device_lenght/(self.periodic_length*1e-3))*self.avg_current*QnK(F_n_k(float(self.deflection_parameter_K),_n_harmonics[i]), float(self.deflection_parameter_K), _n_harmonics[i]))
            
            #Brigth
            _n_brigth = np.append(_n_brigth, _n_flux[i]/(4*np.pi**2*_undu_size_X*1e3*_undu_size_Y*1e3*_undu_diver_X*1e3*_undu_diver_Y*1e3))

        formatting_function = np.vectorize(lambda f: format(f, '6.3e'))
        _summary = pd.DataFrame({
                                'Harmonic': _n_harmonics,
                                'Wavelengths [m]': formatting_function(_n_wavelengths),
                                'Energy [keV]': formatting_function(_n_energys),
                                'Flux [ph/s]': formatting_function(_n_flux),
                                'Brightness [ph/s/mm²/mrad²/0.1%BW]': formatting_function(_n_brigth)
                                })
        cols = _summary.columns.tolist()
        cols = cols[-2:] + cols[:-2]

        dataframe = _summary[::2]
        max_rows=10

        return html.Table([
            html.Thead(
                html.Tr([html.Th(col) for col in dataframe.columns])
            ),
            html.Tbody([
                html.Tr([
                    html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
                ]) for i in range(min(len(dataframe), max_rows))
            ])
        ])


    def type_plot(self, value='Bright', ran=[0,20], typ='log'):
        '''Plot Bright or Flux as a fuction of photon energy'''
        plt.figure(figsize=(5,2))
        _K_array = np.array([5.89, 5.59, 5.30, 5.00, 4.71, 4.42, 4.12, 3.83, 3.53, 3.24, 2.94, 2.65, 2.36, 2.06, 1.77, 1.47, 1.18, 0.88, 0.59, 0.29])
        _odd_harmonics = np.array([1, 3, 5, 7, 9, 11, 13, 15])
        _odd_waves = np.array([])
        _odd_energies = np.array([])
        _odd_flux = np.array([])
        _odd_bright = np.array([])

        F_n_k = lambda k, n:((k**2*n**2)/(1 + k**2/2)**2)*(jv(((n-1)/2), ((n*k**2)/(4+2*k**2))) - jv(((n+1)/2), (n*k**2)/(4+2*k**2)))**2 #Bessel Special Function

        QnK = lambda FnK, k, n:((1 + (k**2/2))/n)*FnK #Bessel Special Function

        for i in range(len(_odd_harmonics)):
            _n_wavelengths, _n_energys = [], np.array([])
            _n_flux, _n_brigth = np.array([]), np.array([])
            
            for j in range(len(_K_array)):
                #Wavelength of nth harmonic
                _n_wavelengths.append((self.periodic_length/(2*_odd_harmonics[i]*self.relative_energy**2))*(1+(_K_array[j]**2/2)+self.relative_energy**2*(0))*(1e-3))
            
                #Photon sizes
                _undu_size_X = np.sqrt(self.source_size_x**2 + (np.sqrt(_n_wavelengths[j]*self.device_lenght)/(4*np.pi))**2)
                _undu_size_Y = np.sqrt(self.source_size_y**2 + (np.sqrt(_n_wavelengths[j]*self.device_lenght)/(4*np.pi))**2)
                
                #Photon divergence
                _undu_diver_X = np.sqrt(self.source_divergence_x**2 + (np.sqrt(_n_wavelengths[j]/self.device_lenght))**2)
                _undu_diver_Y = np.sqrt(self.source_divergence_y**2 + (np.sqrt(_n_wavelengths[j]/self.device_lenght))**2)
                
                #Photon energy of nth harmonic
                _n_energys = np.append(_n_energys, (9.5*_odd_harmonics[i]*self.electron_energy**2)/(self.periodic_length*(1+(_K_array[j]**2/2)+self.relative_energy**2*(0))))
                
                #Flux
                _n_flux = np.append(_n_flux, 1.43e14*(self.device_lenght/(self.periodic_length*1e-3))*self.avg_current*QnK(F_n_k(float(_K_array[j]),_odd_harmonics[i]), float(_K_array[j]), _odd_harmonics[i]))
                
                #Brigth
                _n_brigth = np.append(_n_brigth, _n_flux[j]/(4*np.pi**2*_undu_size_X*1e3*_undu_size_Y*1e3*_undu_diver_X*1e3*_undu_diver_Y*1e3))

            _odd_waves = np.append(_odd_waves, _n_wavelengths)
            _odd_energies = np.append(_odd_energies, _n_energys)
            _odd_flux = np.append(_odd_flux, _n_flux)
            _odd_bright = np.append(_odd_bright, _n_brigth)


        #Arrays splited in 8 sets
        _odd_energies_splited = np.array_split(_odd_energies, 8)
        _odd_flux_splited = np.array_split(_odd_flux, 8)
        _odd_bright_splited = np.array_split(_odd_bright, 8)


        #Creating effective plot:
        if value == 'Flux':
            array = _odd_flux_splited
            texto = 'Photon Flux [photons/ s /0.1% BW]'
        elif value == 'Bright':
            texto = 'Brightness [ph/s/mm²/mrad²/0.1%BW]'
            array = _odd_bright_splited

        harms_legs = ['1º harm', '3º harm', '5º harm', '7º harm', '9º harm', '11º harm',
                      '13º harm', '15º harm']

        fig = go.Figure()
        for i in range(8):
            # Plotly plot
            fig.add_trace(go.Scatter(x=_odd_energies_splited[i], y=array[i],
                                 mode='lines+markers',
                                 name=harms_legs[i]))

            # Matplotlib plot for save
            plt.plot(_odd_energies_splited[i], array[i], '-o', label=harms_legs[i])
        plt.title("The " +str(value)+" as a function of photon energy")
        plt.xlabel('Photon energy (KeV)')
        plt.ylabel(texto)
        plt.grid('True', alpha=0.3)
        plt.yscale(str(typ))

        file_save = plt.savefig(os.path.dirname(os.path.abspath(__file__))+"//images//graph.png")
            
        fig.update_layout(
        title=go.layout.Title(
            text="The " +str(value)+" as a function of photon energy",
            xref="paper",
            x=0.5
        ),
        xaxis=go.layout.XAxis(
            title=go.layout.xaxis.Title(
                text="Photon Energy [KeV]"                
            )
        ),
        yaxis=go.layout.YAxis(
            title=go.layout.yaxis.Title(
                text=texto
            )
        )
        ),
        fig.update_layout(
            yaxis = dict(
                type = str(typ),
                showexponent = 'all',
                exponentformat = 'E'
                ),
            xaxis = dict(
                type='linear',
                range=ran
                )
            )

        return fig

    def beam_stay_clear(self, A_B, axis):
        self._s = np.arange(0,3,0.1)              #Electron positions from center
        Bx = np.array([17.779, 1.357, 1.357])     #BetaX = Beta oscillations from X axis
        By = np.array([3.566, 1.600])             #BetaY = Beta oscillations from Y axis
        BSCx = np.array([11.5, 3.2])              #BSCx = Beam Stay-Clear from X
        BSCy = np.array([2.9, 1.9])               #BSCy = Beam Stay-Clear from Y

        Beta_s = lambda b, s: b+(s**2/b)
        BCS_s = lambda bsc, b, Beta_s: np.sqrt((Beta_s/b))*bsc

        result_beta_s = np.array([])
        result_BCS = np.array([])
        if (A_B == 'A') and (axis == 'x'):   #High Beta and X-axis
            beta_zero = Bx[0]
            BStayClear = BSCx[0]
            
        elif (A_B == 'B') and (axis == 'x'): #Low Beta and X-axis
            beta_zero = Bx[1]
            BStayClear = BSCx[1]
            
        elif (A_B == 'A') and (axis == 'y'): #High Beta and Y-axis
            beta_zero = By[0]
            BStayClear = BSCy[0]
        
        elif (A_B == 'B') and (axis == 'y'): #Low Beta and Y-axis
            beta_zero = By[1]
            BStayClear = BSCy[1]
         
        for i in range(len(self._s)):
                result_beta_s = np.append(result_beta_s, Beta_s(beta_zero, self._s[i]))
                result_BCS = np.append(result_BCS, BCS_s(BStayClear, beta_zero, result_beta_s[i]))
            
        return result_BCS

    def plot_BSC(self):
        fig = make_subplots(rows=2, cols=2, subplot_titles=("High Beta SS", "High Beta SS", "Low Beta SS", "Low Beta SS")
                            )
        self._s = np.arange(0,3,0.1)
        A_B = ['A', 'B']
        axis = ['x', 'y']
        colors = ['darkblue', 'salmon']
        Y_names = ['Horizontal', 'Vertical']
        line_names = ['High Beta', 'Low Beta']
        
        for i in range(1,3): #n row
            for j in range(1,3): #n column
                fig.add_trace(
                    go.Scatter(x=self._s, y=self.beam_stay_clear(A_B[i-1], axis[j-1]), marker_color=colors[i-1], name=line_names[i-1]),
                               row=i, col=j)
                fig.update_xaxes(title_text="Electron positions from center [m]", row=i, col=j)
                fig.update_yaxes(title_text=Y_names[j-1]+" BSC [mm]", showgrid=True, zeroline=True, row=i, col=j)

        # Update fig height
        fig.update_layout(height=800)

        return fig

    def report_mode(self, inst):
        if isinstance(inst, object):
            del(inst)
            
        _rr = reportfiles.ReportCreator()
        
        _rr.machine_variables(self.electron_energy, self.avg_current,
                              self.Circum, self.Bunches, self.sigma_z,
                              self.nat_emittance, self.coupling_cte,
                              self.energy_spread, self.beta_x,
                              self.beta_y, self.eta_x, self.eta_y)

        _rr.str_ring_cal(self.relative_energy, self.emission_angle,
                         self.emittance_horizontal_x, self.emittance_vertical_y,
                         self.source_size_x, self.source_size_y,
                         self.source_divergence_x, self.source_divergence_y)

        _rr.undulator_enter(self.gap_value, self.magnetic_field,
                            self.periodic_length, self.device_lenght)

        _rr.undulator_results(self.deflection_parameter_K, self.wavelength_first_harm, self.first_photon_energy,
                              self.power_undulator, self.angular_power, self.undulator_spectral_flux, self.first_photon_flux,
                              self.undu_size_n1, self.undu_divergence_n1, self.effec_size_x, self.effec_size_y, self.effec_diver_x,
                              self.effec_diver_y, self.bright_undulator_n1)

        import glob
        directory=os.path.dirname(os.path.abspath(__file__))
        os.chdir(directory)
        files=glob.glob('*.pdf')
        for filename in files:
            os.unlink(filename)

        _rr.create_pdf()
