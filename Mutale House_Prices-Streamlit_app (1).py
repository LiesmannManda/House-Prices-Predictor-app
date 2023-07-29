import streamlit as st
from PIL import Image

# Load the image
image = Image.open('/Users/Tale/Desktop/Manda_House_logo.png')


import io
import base64

buffered = io.BytesIO()
image.save(buffered, format="PNG")
img_str = base64.b64encode(buffered.getvalue()).decode()

#logo
st.markdown(
    f'<div style="text-align: center"><img src="data:image/png;base64,{img_str}" width="500"></div>',
    unsafe_allow_html=True,
)




#welcome message
st.markdown("""
<div style="text-align: center">

### Welcome to our House Price Predictor

we have developed multiple models to accurately predict house prices based on critical features of a property. 

The models consider a variety of parameters, including the overall area of the property, the total square footage of the basement, the number of bedrooms, and the capacity of the garage. These factors collectively provide a comprehensive understanding of the property's value.

This tool is designed to offer a reliable prediction of a house's price, assisting both buyers and sellers in making informed decisions in the real estate market. We invite you to input the details of your property and discover its estimated market value.

</div>
""", unsafe_allow_html=True)



import streamlit as st

#video
st.video("/Users/Tale/Desktop/streamlitvideo2.mp4")







import streamlit as st
import pandas as pd
import pickle

#trained model
model = pickle.load(open('/Users/Tale/Desktop/house_prices_trained_pipe_streamlit.sav', 'rb'))


# Instructions
st.markdown("""
<div style="text-align: center">

### Please input the parameters of the house to get an estimated price:

</div>
""", unsafe_allow_html=True)



LotArea = st.number_input("Lot Area (in sq ft)", value=1000)
TotalBsmtSF = st.number_input("Total Basement Area (in sq ft)", value=500)
BedroomAbvGr = st.number_input("Number of Bedrooms", value=2)
GarageCars = st.number_input("Garage Size (in car capacity)", value=1)

# Button to make predictions
if st.button('Predict Price'):

    #dataframe
    new_house = pd.DataFrame({
        'LotArea': [LotArea],
        'TotalBsmtSF': [TotalBsmtSF],
        'BedroomAbvGr': [BedroomAbvGr],
        'GarageCars': [GarageCars]
    })

    #prediction
    prediction = model.predict(new_house)

    #Display the prediction
    st.write("The estimated price of the house is: $", round(float(prediction[0]), 2))
    
    
    # Disclaimer
st.markdown("""
The predicted house price provided by this app is an estimate and should be used for informational purposes only. 
Actual property prices may vary and are subject to market conditions and other factors.

""")
    
import streamlit as st
import pandas as pd
import plotly.express as px

#DataFrame with house prices and number of bedrooms
data = pd.DataFrame({
    'BedroomAbvGr': [3, 4, 3, 5, 2, 4, 3, 4, 3, 2],
    'SalePrice': [250000, 300000, 220000, 280000, 200000, 320000, 230000, 270000, 290000, 210000]
})

#Background colour
st.markdown(
    """
    <style>
    body {
        background-color: navy;  /* Navy color for the app background */
    }
    .stMap {
        background-color: navy;  /* Navy color for the map background */
    }
    .stTitle {
        text-align: center; /* Center the title */
    }
    .stMarkdown {
        font-size: 16px; /* Adjust font size for the text */
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.title("Bedrooms vs House Prices")

#scatter plot showing the relationship between number of bedrooms and house prices
fig = px.scatter(data, x='BedroomAbvGr', y='SalePrice', 
                 labels={'BedroomAbvGr': 'Number of Bedrooms', 'SalePrice': 'Sale Price'}, 
                 hover_name=data.index, hover_data={'BedroomAbvGr': True, 'SalePrice': True})
fig.update_traces(marker=dict(color='blue', size=10), selector=dict(mode='markers'))
fig.update_layout(xaxis_title='Number of Bedrooms', yaxis_title='Sale Price', title=None)
st.plotly_chart(fig)

    
    
    








    
import streamlit as st

# Closing statement
st.write("""
### Thank You for Exploring House Price Predictor!

We hope you enjoyed this magical journey into the world of real estate predictions. The House Price Predictor app offers valuable insights into the estimated prices of houses based on key features. Whether you're a buyer or a seller, this tool can assist you in making informed decisions in the real estate market.

Remember, the predictions provided by this app are estimates and should be used as a reference. The true value of a property depends on various factors, and consulting a real estate professional is always recommended.

If you have any questions or feedback, feel free to reach out to us:

- Email: info@MandaHousepredictor.co.uk
- Phone: 01424123123

Happy house hunting and best wishes for your real estate ventures!
""")




    