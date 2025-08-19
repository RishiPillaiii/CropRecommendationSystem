from flask import Flask, request, render_template
import numpy as np
import pickle

# Load model and scalers
model = pickle.load(open('models/model.pkl', 'rb'))
sc = pickle.load(open('models/standscaler.pkl', 'rb'))
ms = pickle.load(open('models/minmaxscaler.pkl', 'rb'))

# Create Flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/contact')
def contact():
    return render_template("contact.html")


@app.route('/search', methods=['GET', 'POST'])
def search():
    if request.method == 'POST':
        crop_name = request.form['crop_name'].strip().capitalize()

        crop_list = [
            "Rice", "Maize", "Jute", "Cotton", "Coconut", "Papaya", "Orange",
            "Apple", "Muskmelon", "Watermelon", "Grapes", "Mango", "Banana",
            "Pomegranate", "Lentil", "Blackgram", "Mungbean", "Mothbeans",
            "Pigeonpeas", "Kidneybeans", "Chickpea", "Coffee"
        ]

        if crop_name in crop_list:
            result = f"✅ {crop_name} is available in our recommendation list."
        else:
            result = f"❌ {crop_name} is not found in our recommendation list."

        return render_template('search.html', result=result)

    return render_template('search.html')



@app.route("/predict", methods=['POST'])
def predict():
    try:
        # Convert all inputs to float
        N = float(request.form['Nitrogen'])
        P = float(request.form['Phosporus'])
        K = float(request.form['Potassium'])
        temp = float(request.form['Temperature'])
        humidity = float(request.form['Humidity'])
        ph = float(request.form['Ph'])
        rainfall = float(request.form['Rainfall'])

        # Prepare features
        feature_list = [N, P, K, temp, humidity, ph, rainfall]
        single_pred = np.array(feature_list).reshape(1, -1)

        # Apply scaling
        scaled_features = ms.transform(single_pred)
        final_features = sc.transform(scaled_features)

        # Make prediction
        prediction = model.predict(final_features)

        crop_dict = {
            1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut",
            6: "Papaya", 7: "Orange", 8: "Apple", 9: "Muskmelon",
            10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
            14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean",
            18: "Mothbeans", 19: "Pigeonpeas", 20: "Kidneybeans",
            21: "Chickpea", 22: "Coffee"
        }

        if prediction[0] in crop_dict:
            crop = crop_dict[prediction[0]]
            result = f"{crop} is the best crop to be cultivated right there."
        else:
            result = "Sorry, we could not determine the best crop to be cultivated with the provided data."

        return render_template('index.html', result=result)

    except Exception as e:
        # Show error message in UI if something fails
        return render_template('index.html', result=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
