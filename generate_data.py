import numpy as np
import pandas as pd

def main():
    random_generator = np.random.default_rng()
    observations = 250

    temperature = random_generator.normal(25, 4, observations)
    humidity = random_generator.normal(50, 10, observations)
    time = random_generator.normal(10, 2, observations)
    pressure = random_generator.normal(100, 8, observations)
    speed = random_generator.normal(20, 5, observations)
    concentration = random_generator.normal(5, 1, observations)
    vibration = random_generator.normal(15, 3, observations)
    material_quantity = random_generator.normal(30, 6, observations)
    random_error = random_generator.normal(0, 3, observations)
    quality = (100 + 1.8 * (temperature - 25) - 0.7 * (humidity - 50)
        + 1.2 * (time - 10) + random_error)

    data = pd.DataFrame({
        "Temperatura": temperature,
        "Humedad": humidity,
        "Tiempo": time,
        "Presion": pressure,
        "Velocidad": speed,
        "Concentracion": concentration,
        "Vibracion": vibration,
        "CantidadMaterial": material_quantity,
        "Calidad": quality
    })

    data.to_csv("datos.csv", index=False)
    print("Archivo datos.csv creado correctamente.")
    print(f"Observaciones: {len(data)}")
    print(f"Variables predictoras: {len(data.columns) - 1}")
    print("Variable objetivo: Calidad")

if __name__ == "__main__":
    main()