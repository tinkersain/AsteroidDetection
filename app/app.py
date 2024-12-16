import streamlit as st
import pickle
import numpy as np
import os

# Load the model
try:
    with open('finalModel.pkl', 'rb') as file:
        model = pickle.load(file)
    print("Model loaded successfully.")
except FileNotFoundError:
    print(f"Error: The file was not found at {os.getcwd()}")
except Exception as e:
    print(f"An error occurred: {e}")


orbit_classification = {
    "AMO": 1, "APO": 2, "AST": 3, "ATE": 4, 
    "CEN": 5, "HYA": 6, "IEO": 7, "IMB": 8,
    "MBA": 9, "MCA": 10, "OMB": 11, "TJN": 12,
    "TNO": 13
}

# Streamlit UI setup
st.title("Asteroid Hazard Prediction")
st.write("Input the features of an asteroid to predict if it is potentially hazardous.")


# Initialize session state for button text
if 'btnTxt' not in st.session_state:
    st.session_state.btnTxt = "Hazardous Values"
    st.session_state.btnColor = "red"


# Custom CSS for positioning the button
st.markdown(f"""
    <style>
        .top-right-button {{
            position: fixed;
            background-color: {st.session_state.btnColor}
            color: white
            top: 20px;
            right: 20px;
            z-index: 10;
        }}
    </style>
""", unsafe_allow_html=True)

# Function to be called when the button is clicked
def my_function():
    # Toggle button text
    if st.session_state.btnTxt == "Hazardous Values":
        st.session_state.btnTxt = "Non Hazardous Values"
        st.session_state.btnColor = "green"
    else:
        st.session_state.btnTxt = "Hazardous Values"
        st.session_state.btnColor = "red"
    st.experimental_rerun()

# Button in the top-right corner
# if st.button(st.session_state.btnTxt, key="top_right_button", help="Click to trigger the function",use_container_width=True):
#     my_function()

# Display the button with updated text immediately after click

# values = values = [
#     18.0, 50.00, 0.25, 0.1, 2459200.5, 59200.0, 20210425.0, 0.8, 1.0, 0.1, 
#     10.0, 180.0, 90.0, 0.0, 1.8, 0.9, 2459000.5, 20210101.0, 300.0, 0.8, 
#     0.01, 2.0, 0.001, 0.0001, 0.0001, 0.01, 0.01, 0.01, 0.01, 0.0001, 
#     0.00001, 0.1, 0.1, 0.05, 1, 0.03
# ]


values = [
    22.0, 0.5, 0.15, 0.05, 2459200.5, 59200.0, 20210425.0, 0.2, 1.5, 0.9, 
    10.0, 180.0, 90.0, 0.0, 2.1, 0.5, 2459000.5, 20210101.0, 365.25, 1.0, 
    0.05, 19.5, 0.001, 0.0001, 0.0001, 0.01, 0.01, 0.01, 0.01, 0.0001, 
    0.00001, 0.1, 0.1, 0.1, 0
]





# Input fields for the features
# Input fields for the features
H = st.number_input("H (Absolute Magnitude)", min_value=0.0, step=0.1, value=values[0], help="Brightness of the asteroid")
Diameter = st.number_input("Diameter (km)", min_value=0.0, step=0.1, value=values[1], help="Estimated diameter in kilometers")
Albedo = st.number_input("Albedo (Reflectivity)", min_value=0.0, step=0.01, value=values[2], help="Reflectivity of the asteroid's surface")
Diameter_sigma = st.number_input("Diameter Sigma (km)", min_value=0.0, step=0.01, value=values[3], help="Uncertainty in the diameter")
Epoch = st.number_input("Epoch (Julian Days)", min_value=0.0, step=0.1, value=values[4], help="Time for orbital calculation")
Epoch_mjd = st.number_input("Epoch MJD (Modified Julian Date)", min_value=0.0, step=0.1, value=values[5], help="Modified Julian Date")
Epoch_cal = st.number_input("Epoch Calendar (Date Format)", min_value=0.0, step=0.1, value=values[6], help="Calendar format epoch")
e = st.number_input("e (Eccentricity)", min_value=0.0, step=0.001, value=values[7], help="Orbital eccentricity")
a = st.number_input("a (Semi-Major Axis in AU)", min_value=0.0, step=0.001, value=values[8], help="Semi-major axis in astronomical units")
q = st.number_input("q (Perihelion Distance in AU)", min_value=0.0, step=0.001, value=values[9], help="Closest distance to the Sun")
i = st.number_input("i (Inclination in Degrees)", min_value=0.0, step=0.1, value=values[10], help="Inclination with respect to the ecliptic plane")
om = st.number_input("om (Longitude of Ascending Node in Degrees)", min_value=0.0, step=0.1, value=values[11], help="Location of ascending node")
w = st.number_input("w (Argument of Perihelion in Degrees)", min_value=0.0, step=0.1, value=values[12], help="Orientation of perihelion")
ma = st.number_input("ma (Mean Anomaly in Degrees)", min_value=0.0, step=0.1, value=values[13], help="Position along orbit at epoch")
ad = st.number_input("ad (Aphelion Distance in AU)", min_value=0.0, step=0.001, value=values[14], help="Farthest distance from the Sun")
n = st.number_input("n (Mean Motion in Degrees/Day)", min_value=0.0, step=0.01, value=values[15], help="Orbital speed")
tp = st.number_input("tp (Time of Perihelion Passage in Julian Days)", min_value=0.0, step=0.1, value=values[16], help="Time of closest approach to the Sun")
tp_cal = st.number_input("tp_cal (Time of Perihelion Passage in Calendar Date)", min_value=0.0, step=0.1, value=values[17], help="Calendar format time of perihelion passage")
per = st.number_input("per (Orbital Period in Days)", min_value=0.0, step=0.1, value=values[18], help="Time for one orbit completion")
per_y = st.number_input("per_y (Orbital Period in Years)", min_value=0.0, step=0.01, value=values[19], help="Orbital period in years")
moid = st.number_input("moid (MOID in AU)", min_value=0.0, step=0.001, value=values[20], help="Minimum Orbit Intersection Distance")
moid_ld = st.number_input("moid_ld (MOID in Lunar Distances)", min_value=0.0, step=0.01, value=values[21], help="MOID in Earth-Moon distances")
sigma_e = st.number_input("sigma_e (Uncertainty in Eccentricity)", min_value=0.0, step=0.001, value=values[22], help="Uncertainty in orbital eccentricity")
sigma_a = st.number_input("sigma_a (Uncertainty in Semi-Major Axis in AU)", min_value=0.0, step=0.001, value=values[23], help="Uncertainty in semi-major axis")
sigma_q = st.number_input("sigma_q (Uncertainty in Perihelion Distance in AU)", min_value=0.0, step=0.001, value=values[24], help="Uncertainty in perihelion distance")
sigma_i = st.number_input("sigma_i (Uncertainty in Inclination in Degrees)", min_value=0.0, step=0.001, value=values[25], help="Uncertainty in orbital inclination")
sigma_om = st.number_input("sigma_om (Uncertainty in Longitude of Ascending Node in Degrees)", min_value=0.0, step=0.001, value=values[26], help="Uncertainty in longitude of ascending node")
sigma_w = st.number_input("sigma_w (Uncertainty in Argument of Perihelion in Degrees)", min_value=0.0, step=0.001, value=values[27], help="Uncertainty in argument of perihelion")
sigma_ma = st.number_input("sigma_ma (Uncertainty in Mean Anomaly in Degrees)", min_value=0.0, step=0.001, value=values[28], help="Uncertainty in mean anomaly")
sigma_ad = st.number_input("sigma_ad (Uncertainty in Aphelion Distance in AU)", min_value=0.0, step=0.001, value=values[29], help="Uncertainty in aphelion distance")
sigma_n = st.number_input("sigma_n (Uncertainty in Mean Motion in Degrees/Day)", min_value=0.0, step=0.001, value=values[30], help="Uncertainty in mean motion")
sigma_tp = st.number_input("sigma_tp (Uncertainty in Time of Perihelion Passage in Julian Days)", min_value=0.0, step=0.1, value=values[31], help="Uncertainty in time of perihelion passage")
sigma_per = st.number_input("sigma_per (Uncertainty in Orbital Period in Days)", min_value=0.0, step=0.1, value=values[32], help="Uncertainty in orbital period")
rms = st.number_input("rms (Root Mean Square of Residuals)", min_value=0.0, step=0.001, value=values[33], help="Fit of observed positions to the orbital model")
neo = st.radio("NEO (Near-Earth Object Flag)", options=[0, 1], index=values[34], help="Boolean flag indicating if the asteroid is a Near-Earth Object")
classification_select = st.selectbox("Class (Orbit Classification)", options=list(orbit_classification.keys()), index=0, help="Classification of the asteroid based on its orbit")
classification = selected_value = orbit_classification[classification_select]


# Combine all inputs into a feature array
features = np.array([
    H, Diameter, Albedo, Diameter_sigma, Epoch, Epoch_mjd, Epoch_cal, e, a, q, i, om, w, ma, ad, n,
    tp, tp_cal, per, per_y, moid, moid_ld, sigma_e, sigma_a, sigma_q, sigma_i, sigma_om, sigma_w,
    sigma_ma, sigma_ad, sigma_n, sigma_tp, sigma_per, rms, neo, classification
]).reshape(1, -1)

# Predict using the model
if st.button("Predict"):
    prediction = model.predict(features)
    if prediction[0] == 1:
        st.success("The asteroid is potentially hazardous.")
    else:
        st.success("The asteroid is NOT potentially hazardous.")
