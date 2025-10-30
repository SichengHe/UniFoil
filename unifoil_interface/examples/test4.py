import unifoil.extract_data as ed

extractor = ed.ExtractData()
x, y = extractor.extract_airfoil_coords(airfoil_number=10004, source="turb", plot_flag=True)
