import pandas as pd


class Conversor:
    def __init__(self):
        self.__data_frame = None
        self.__outlers = {
            'G': {'flt': [0.06129732355475424, 0.27289628610014904], 'lds': [-22.32607102394104, -1.3311007022857666],
                  'strpk': [0.1641031801700592, 1.8732363283634186], 'flu': [0.04072224255651239, 0.12555399443954215],
                  'entr': [6.43141233921051, 8.202026009559631], 'danc': [0.7248491942882538, 1.9346024692058563],
                  'bpm': [65.28873348236084, 181.0199728012085], 'ptch': [0.3889099434018135, 0.6416881754994392],
                  'mfcc1': [-755.3796287500002, -560.6806187499999], 'mfcc2': [50.95847625000002, 187.92795425],
                  'mfcc3': [-40.761771124999996, 55.675407875], 'mfcc4': [-8.994660624999998, 52.999048375]},
            'NG': {'alds': [0.8655174970626831, 1.0372579097747803, 0.9295764177389566],
                   'nrg': [0.0016927169635891498, 0.056420412845909554, 0.030583320065380705]}
        }
        self.__escalar = {
            'G': {'lds': [-12.00908887, 3.68078636], 'nrg': [0.02915877, 0.0095662],
                  'flu': [0.08300337, 0.01653184], 'bpm': [122.91790398, 22.33771277],
                  'ptch': [0.51488477, 0.04888783], 'mfcc3': [7.11287377, 19.10214552]},
            'NG': {'flt': [0.06975416, 0.27421737], 'alds': [0.8681559, 0.99068403],
                   'strpk': [0.15187809, 1.87806413], 'entr': [6.4390043, 8.19057977],
                   'danc': [0.72345431, 1.93494983], 'mfcc1': [-756.60627375, -559.26528375],
                   'mfcc2': [50.49788188, 188.33292687], 'mfcc4': [-8.70278937, 52.49991762]}
        }

    def analizar_outlers(self):
        columnas_gaussianas = self.__outlers['G'].keys()
        columnas_no_gaussianas = self.__outlers['NG'].keys()
        for i in columnas_gaussianas:
            valor_temp = self.__data_frame[i].values
            valor_temp = valor_temp[0]
            if valor_temp < self.__outlers['G'][i][0]:
                self.__data_frame[i] = self.__outlers['G'][i][0]
            elif valor_temp > self.__outlers['G'][i][1]:
                self.__data_frame[i] = self.__outlers['G'][i][1]
        for i in columnas_no_gaussianas:
            valor_temp = self.__data_frame[i].values
            valor_temp = valor_temp[0]
            if valor_temp < self.__outlers['NG'][i][0] or valor_temp > self.__outlers['NG'][i][1]:
                self.__data_frame[i] = self.__outlers['NG'][i][2]

    def escalar_valores(self):
        columnas_gaussianas = self.__escalar['G'].keys()
        columnas_no_gaussianas = self.__escalar['NG'].keys()
        for i in columnas_gaussianas:
            valor_temp = self.__data_frame[i].values
            valor_temp = valor_temp[0]
            self.__data_frame[i] = (valor_temp - self.__escalar['G'][i][0]) / self.__escalar['G'][i][1]
        for i in columnas_no_gaussianas:
            valor_temp = self.__data_frame[i].values
            valor_temp = valor_temp[0]
            self.__data_frame[i] = (valor_temp - self.__escalar['NG'][i][0]) / (
                        self.__escalar['NG'][i][1] - self.__escalar['NG'][i][0])

    def transformar_entrada(self, _data_frame):
        sin_nombre = _data_frame.drop(['Name'], axis=1)
        self.set_dataframe(sin_nombre)
        self.analizar_outlers()
        self.escalar_valores()
        self.__data_frame.insert(0, "Name", _data_frame["Name"].values[0])

    def set_dataframe(self, _data_frame):
        self.__data_frame = _data_frame

    def get_dataframe(self):
        return self.__data_frame


if __name__ == '__main__':
    columnas = ['Name', 'flt', 'lds', 'alds', 'strpk', 'nrg', 'flu', 'entr', 'danc', 'bpm',
                'ptch', 'mfcc1', 'mfcc2', 'mfcc3', 'mfcc4']
    datosPrueba = [["olvidala", 0.240082, -14.933464, 0.897602, 1.146760, 0.022006, 0.071873, 7.002771, 1.288145, 114.862709,
                    0.419456, -723.15356, 104.381910, 13.517786, 49.038660]]
    a = Conversor()
    data_limpia = pd.DataFrame(columns=columnas)
    data = pd.read_csv("dataClean/FeaturesObtain.csv", sep=',', header='infer')
    data = data[columnas]
    for i in range(len(data)):
        fila_temp = data.iloc[i]
        fila_temp = pd.DataFrame(fila_temp).transpose()
        a.transformar_entrada(fila_temp)
        fila_temp = a.get_dataframe()
        data_limpia = data_limpia.append(fila_temp,ignore_index=True)
    data_limpia.to_csv('dataCleanFinalConNombres', index=False)
