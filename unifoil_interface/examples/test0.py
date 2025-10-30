import unifoil
unifoil.gen_ft()
unifoil.gen_nlf()

from unifoil.extract_data import ExtractData
ed = ExtractData()
x, y = ed.extract_airfoil_coords(airfoil_number=3, source="turb", plot_flag=True)

x, y = ed.extract_airfoil_coords(airfoil_number=3, source="translam", plot_flag=True)