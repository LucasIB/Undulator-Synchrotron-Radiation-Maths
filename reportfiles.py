import os
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate  #Table automatically in the center
from reportlab.lib.pagesizes import letter
from reportlab.platypus import Table
from time import gmtime, strftime
import pandas as pd
import numpy as np

class ReportCreator(object):
    def __init__(self):
        '''Requirements''' 
        self.filename = " my_undulator_report.pdf"
        self.documentTitle = "Undulator pameters Calculus"
        self.title = 'Report Undulator Parameters Maths'
        self.subTitle = ''
        self.textLine = [
            'Synchrotron Radiation parameters from undulator light source'
            ]
        self.text_1 = ['1.0 - SIRUS Storage Ring Specification Parameters']

        self.text_2 = ['2.0 - Storage Ring Parameters Calculation']

        self.text_3 = ['3.0 - Undulator Input Parameters']

        self.text_4 = ['4.0 - Undulator Radiation Output (theta=0)']

        self.image = 'LNLS_sem_fundo.png'
        
        self.outfilepath = os.path.join(os.path.dirname(os.path.abspath(__file__))+'//external_file//'+self.datetime()+self.filename)
        self.pdf = canvas.Canvas(self.outfilepath)
          
        
    def machine_variables(self, electron_energy, avg_current, Circum, Bunches, sigma_z,
                       nat_emittance, coupling_cte, energy_spread, beta_x, beta_y,
                       eta_x, eta_y):
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

        self._list_of_lists_variables = [
            ['Electron Energy', str(self.electron_energy), 'Average Current (A)', str(self.avg_current)],
            ['Circumference (m)', str(self.Circum), 'Bunches', str(self.Bunches)],
            ['Bunch length (mm)', str(self.sigma_z), 'Nat. Emittance (m.rad)', '{:.2e}'.format(self.nat_emittance)],
            ['Coupling Constant', str(self.coupling_cte), 'Energy Spread', '{:.2e}'.format(self.energy_spread)],
            ['Beta x (m)', str(self.beta_x), 'Beta y (m)', str(self.beta_y)],
            ['Eta x (m)', str(self.eta_x), 'Eta y (m)', str(self.eta_y)]
            ]

    def str_ring_cal(self, relative_energy, emission_angle, emittance_horizontal_x,
                     emittance_vertical_y, source_size_x, source_size_y, source_divergence_x,
                     source_divergence_y):
        #Storage ring calculus
        self.relative_energy = relative_energy
        self.emission_angle = emission_angle
        self.emittance_horizontal_x = emittance_horizontal_x
        self.emittance_vertical_y = emittance_vertical_y
        self.source_size_x = source_size_x
        self.source_size_y = source_size_y
        self.source_divergence_x = source_divergence_x
        self.source_divergence_y = source_divergence_y

        self._list_of_lists_strcalcs = [
            ['Relative Energy (eV)', str(self.relative_energy), 'Emission angle (rad)', '{:.2e}'.format(self.emission_angle)],
            ['Emittance horizontal x (m.rad)', '{:.2e}'.format(self.emittance_horizontal_x), 'Emittance vertical y (m.rad)', '{:.2e}'.format(self.emittance_vertical_y)],
            ['Source Size x (m)', '{:.2e}'.format(self.source_size_x), 'Source Size y (m)', '{:.2e}'.format(self.source_size_y)],
            ['Source divergence x (rad)', '{:.2e}'.format(self.source_divergence_x), 'Source divergence y (rad)', '{:.2e}'.format(self.source_divergence_y)]
            ]

    def undulator_enter(self, gap_value, magnetic_field, periodic_length, device_lenght):
        # Undulator variables
        self.gap_value = gap_value
        self.magnetic_field = magnetic_field
        self.periodic_length = periodic_length
        self.device_lenght = device_lenght

        self._list_of_lists_undu_enter = [
            ['Gap value (mm)', str(self.gap_value), 'Magnetic field (T)', str(self.magnetic_field)],
            ['Periodic lenght (mm)', str(self.periodic_length), 'Device length (m)', str(self.device_lenght)]
            ]

    def undulator_results(self, deflection_parameter_K, wavelength_first_harm, first_photon_energy,
                          power_undulator, angular_power, undulator_spectral_flux, first_photon_flux,
                          undu_size_n1, undu_divergence_n1, effec_size_x, effec_size_y, effec_diver_x,
                          effec_diver_y, bright_undulator_n1):

        self.deflection_parameter_K = deflection_parameter_K
        self.wavelength_first_harm = wavelength_first_harm

        self._list_of_lists_undu_output = [
            ['Deflection parameter K', '{:.3e}'.format(deflection_parameter_K), 'Undulator source size n=1 m', '{:.3e}'.format(undu_size_n1)],
            ['Wavelenght 1st (theta=0)', '{:.3e}'.format(wavelength_first_harm), 'Undulator source divergence n=1', '{:.3e}'.format(undu_divergence_n1)],
            ['Photon energy of 1st (keV)', '{:.3e}'.format(first_photon_energy), 'Photon beam sizes x, n=1 (m)', '{:.3e}'.format(effec_size_x)],
            ['Total Pwr radiated (kW)', '{:.3e}'.format(power_undulator), 'Photon beam sizes y, n=1 (m)', '{:.3e}'.format(effec_size_y)],
            ['Ang. radiation pwr (W/mrad²)', '{:.3e}'.format(angular_power), 'Photon beam divergence x, n=1 (m)', '{:.3e}'.format(effec_diver_x)],
            ['Ang. Spec Flux (ph/s/mrad²/0.1%BW)', '{:.3e}'.format(undulator_spectral_flux), 'Photon beam divergence y, n=1 (m)', '{:.3e}'.format(effec_diver_y)],
            ['Photon Flux 1st (ph/s/0.1%)', '{:.3e}'.format(first_photon_flux), 'Brightness 1st (ph/s/mrad²/mm²/0.1%BW)', '{:.3e}'.format(bright_undulator_n1)]
            ]
    

    def drawMyRuler(self):
        self.pdf.drawString(100,810, 'x100')
        self.pdf.drawString(200,810, 'x200')
        self.pdf.drawString(300,810, 'x300')
        self.pdf.drawString(400,810, 'x400')
        self.pdf.drawString(500,810, 'x500')

        self.pdf.drawString(10,100, 'y100')
        self.pdf.drawString(10,200, 'y200')
        self.pdf.drawString(10,300, 'y300')
        self.pdf.drawString(10,400, 'y400')
        self.pdf.drawString(10,500, 'y500')
        self.pdf.drawString(10,600, 'y600')
        self.pdf.drawString(10,700, 'y700')
        self.pdf.drawString(10,800, 'y800')

    def datetime(self):
        return strftime("%Y-%m-%d_%H_%M_%S_", gmtime())
        
    def create_pdf(self):
        self.pdf.setTitle(self.documentTitle)              # Document name.pdf
        self.pdf.setFont('Helvetica-Bold', 18)             # Setting font and font size
        self.pdf.drawCentredString(300, 770, self.title)   # Title coordenates X, Y in the middle

        #draw a line
        self.pdf.line(30, 750, 550, 750)

        #draw text
        self.drawtext(self.textLine, 12, 40, 730)

        #draw table 1 header
        self.drawtext(self.text_1, 10, 40, 706) #(font, x, y)

        #draw table 1
        self.drawtable(self._list_of_lists_variables, 40, 590) #(table, x, y)

        #draw table 2 header
        self.drawtext(self.text_2, 10, 40, 560) #(font, x, y)

        #draw table 2
        self.drawtable(self._list_of_lists_strcalcs, 40, 480)

        #draw table 3 header
        self.drawtext(self.text_3, 10, 40, 461) #(font, x, y)

        #draw table 3
        self.drawtable(self._list_of_lists_undu_enter, 40, 415)

        #draw table 4 header
        self.drawtext(self.text_4, 10, 40, 385) #(font, x, y)

        #draw table 4
        self.drawtable(self._list_of_lists_undu_output, 40, 254)

        #draw image
        self.drawimage(os.path.dirname(os.path.abspath(__file__))+'//images//graph.png', 30, 5)

        self.save_file()
    
    def save_file(self):
        self.pdf.showPage()
        self.pdf.save()
        relative_filename = os.path.join(
        'external_file',
        self.datetime() + self.filename)
        return '/{}'.format(relative_filename)

    def drawimage(self, image, x, y):
        return self.pdf.drawInlineImage(image, x, y)

    def drawtext(self, text, font, x, y):
        _text = self.pdf.beginText(x, y)
        _text.setFont('Helvetica', font)
        _text.setFillColor(colors.black)
        for line in text:
            _text.textLine(line)
        return self.pdf.drawText(_text)
        
    def drawtable(self, data, x, y):  #Data must be list of lists
        width = 400
        height = 100
        _pdf = SimpleDocTemplate(
            self.filename,
            pagesize=letter
            )
        _table = Table(data)
        
        from reportlab.platypus import TableStyle
        from reportlab.lib import colors

        style = TableStyle([
            ('ALIGN',(0,0),(-1,-1),'RIGHT'),
        ])
        _table.setStyle(style)

        #Alternating backgroun color
        rowNumb = len(data)
        for i in range(1, rowNumb):
            if i % 2 == 0:
                bc = colors.whitesmoke
            else:
                bc = colors.lavenderblush
            ts = TableStyle(
                [('BACKGROUND', (0,i),(-1,i), bc)]
                )
            _table.setStyle(ts)

        #Add the borders
        ts = TableStyle([
              ('BOX', (0,0),(-1,-1),1,colors.black),
              ('LINEBEFORE', (2,0), (2,-1),1,colors.black),
              ('GRID',(0,0),(-1,-1),0.5,colors.grey)
            ])
        _table.setStyle(ts)

        _elems = []
        _elems.append(_table)
        _table.wrapOn(self.pdf, width, height)
        _table.drawOn(self.pdf, x, y)

