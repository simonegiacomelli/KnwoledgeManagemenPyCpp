import os

import extract_data_cc
import extract_data_py

os.remove("data.csv")

extract_data_py.main()
extract_data_cc.main()
