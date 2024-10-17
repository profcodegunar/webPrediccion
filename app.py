from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Cargar el modelo entrenado
modelo = joblib.load('modelo_ventas_por_año.pkl')

# Ruta para la página de inicio
@app.route('/')
def index():
    return render_template('index.html')

# Ruta para la página "Nosotros"
@app.route('/nosotros')
def nosotros():
    return render_template('nosotros.html')

# Ruta para la página "Simulador de Ventas"
@app.route('/simulador', methods=['GET', 'POST'])
def simulador():
    if request.method == 'POST':
        # Obtener el año del formulario
        año = int(request.form['año'])
        
        # Preparar los datos para el modelo
        datos_para_modelo = np.array([año]).reshape(1, -1)
        
        # Realizar la predicción
        prediccion = modelo.predict(datos_para_modelo)[0]
        
        return render_template('simulador.html', prediccion=prediccion)
    
    return render_template('simulador.html', prediccion=None)

# Ruta para la página "Contactos"
@app.route('/contactos')
def contactos():
    return render_template('contactos.html')

if __name__ == '__main__':
    app.run(debug=True)
