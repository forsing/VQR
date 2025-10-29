
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

from qiskit.visualization import circuit_drawer

from IPython.display import display
from IPython.display import clear_output


from qiskit_machine_learning.utils import algorithm_globals
import random

# =========================
# Seed za reproduktivnost
# =========================
SEED = 35
np.random.seed(SEED)
random.seed(SEED)
algorithm_globals.random_seed = SEED

# 1. Učitaj loto podatke
df = pd.read_csv("loto5_89_k80.csv", header=None)


###################################
print()
print("Prvih 5 ucitanih kombinacija iz CSV fajla:")
print()
print(df.head())
print()
"""
Prvih 5 ucitanih kombinacija iz CSV fajla:

   0   1   2   3   4  5
0  8  10  15  21  31  6
1  7  16  19  25  31  2
2  1   7  18  25  28  7
3  6   7  12  19  22  2
4  8  25  29  33  34  3
"""

print()
print("Zadnjih 5 ucitanih kombinacija iz CSV fajla:")
print()
print(df.tail())
print()
"""
Zadnjih 5 ucitanih kombinacija iz CSV fajla:

    0   1   2   3   4  5
81  2   6  15  17  20  3
82  3   5  20  28  35  9
83  1   7  11  28  31  4
84  8  11  16  22  27  6
85  7   9  13  14  28  2
"""
####################################


# 2. Minimalni i maksimalni dozvoljeni brojevi po poziciji
min_val = [1, 2, 3, 4, 5, 1]
max_val = [31, 32, 33, 34, 35, 10]

# 3. Funkcija za mapiranje brojeva u indeksirani opseg [0..range_size-1]
def map_to_indexed_range(df, min_val, max_val):
    df_indexed = df.copy()
    for i in range(df.shape[1]):
        df_indexed[i] = df[i] - min_val[i]
        # Provera da li su svi brojevi u validnom opsegu
        if not df_indexed[i].between(0, max_val[i] - min_val[i]).all():
            raise ValueError(f"Vrednosti u koloni {i} nisu u opsegu 0 do {max_val[i] - min_val[i]}")
    return df_indexed

# 4. Primeni mapiranje
df_indexed = map_to_indexed_range(df, min_val, max_val)

# 5. Provera rezultata
print()
print(f"Učitano kombinacija: {df.shape[0]}, Broj pozicija: {df.shape[1]}")
print()
"""
Učitano kombinacija: 86, Broj pozicija: 6
"""


print()
print("Prvih 5 mapiranih kombinacija:")
print()
print(df_indexed.head())
print()
"""
Prvih 5 mapiranih kombinacija:

   0   1   2   3   4  5
0  7   8  12  17  26  5
1  6  14  16  21  26  1
2  0   5  15  21  23  6
3  5   5   9  15  17  1
4  7  23  26  29  29  2
"""

print()
print("Zadnjih 5 mapiranih kombinacija:")
print()
print(df_indexed.tail())
print()
"""
Zadnjih 5 mapiranih kombinacija:

    0  1   2   3   4  5
81  1  4  12  13  15  2
82  2  3  17  24  30  8
83  0  5   8  24  26  3
84  7  9  13  18  22  5
85  6  7  10  10  23  1
"""




# Parametri
num_qubits = 5          # 5 qubita po poziciji
num_layers = 2          # Dubina varijacionog sloja
num_positions = 5       # 6 pozicija (brojeva) u loto kombinaciji

def encode_position(value):
    """
    Sigurno enkoduje 'value' u QuantumCircuit sa tacno num_qubits qubita.
    Ako value zahteva vise bitova od num_qubits, koristi se LSB (zadnjih num_qubits bitova),
    i ispisuje se upozorenje.
    """
    # osiguraj int
    v = int(value)
    bin_full = format(v, 'b')  # pravi binarni bez vodećih nula
    if len(bin_full) > num_qubits:
        # upozorenje: vrednost ne staje u broj qubita; koristimo zadnjih num_qubits bita (LSB)
        print(f"Upozorenje: value={v} zahteva {len(bin_full)} bitova, a num_qubits={num_qubits}. Koristim zadnjih {num_qubits} bita.")
        bin_repr = bin_full[-num_qubits:]
    else:
        bin_repr = bin_full.zfill(num_qubits)

    qc = QuantumCircuit(num_qubits)
    # reversed da bi LSB išao na qubit 0 (ako želiš suprotno, ukloni reversed)
    for i, bit in enumerate(reversed(bin_repr)):
        if bit == '1':
            qc.x(i)
    return qc

# Varijacioni sloj: Ry rotacije + CNOT lanac
def variational_layer(params):
    qc = QuantumCircuit(num_qubits)
    for i in range(num_qubits):
        qc.ry(params[i], i)
    for i in range(num_qubits - 1):
        qc.cx(i, i+1)
    return qc

# QCBM ansambl: slojevi varijacionih blokova
def qcbm_ansatz(params):
    qc = QuantumCircuit(num_qubits)
    for layer in range(num_layers):
        start = layer * num_qubits
        end = (layer + 1) * num_qubits
        qc.compose(variational_layer(params[start:end]), inplace=True)
    return qc

# Kompletan QCBM za svih 7 pozicija
def full_qcbm(params_list, values):
    total_qubits = num_qubits * num_positions
    qc = QuantumCircuit(total_qubits)

    for pos in range(num_positions):
        start_q = pos * num_qubits
        end_q = start_q + num_qubits

        # Enkoduj vrednost za poziciju
        qc_enc = encode_position(values[pos])
        qc.compose(qc_enc, qubits=range(start_q, end_q), inplace=True)

        # Dodaj varijacioni ansambl
        qc_var = qcbm_ansatz(params_list[pos])
        qc.compose(qc_var, qubits=range(start_q, end_q), inplace=True)

    # Dodaj merenja za svih 30 qubita
    qc.measure_all()

    return qc


test_values = [27,16,35,34,12,4]
np.random.seed(35)
params_list = [np.random.uniform(0, 2*np.pi, num_layers * num_qubits) for _ in range(num_positions)]

# Generiši QCBM za svih 6 pozicija
full_circuit = full_qcbm(params_list, test_values)



# Prikaz celog kruga u 'mpl' formatu
full_circuit.draw('mpl')
# plt.show()

# fold=40 prelama linije tako da veliki krug stane na ekran.
full_circuit.draw('mpl', fold=40)
# plt.show()


# The only valid choices are 
# text, latex, latex_source, and mpl


# Kompaktni prikaz kola
print("\nKompaktni prikaz kvantnog kola (text):\n")
# print(full_circuit.draw('text'))
"""
Kompaktni prikaz kvantnog kola (text):


"""


# display(full_circuit.draw())     
display(full_circuit.draw("mpl"))
# plt.show()


circuit_drawer(full_circuit, output='latex', style={"backgroundcolor": "#EEEEEE"})
# plt.show()


# import tinytex
# pip install tinycio
# pip install torchvision
# tinytex.install()



"""
# Sačuvaj kao PDF
img1 = full_circuit.draw('latex')
img1.save("/data/qc30_5_1.pdf")


# Sačuvaj kao sliku u latex formatu jpg
img2 = full_circuit.draw('latex')
img2.save("/data/qc30_5_2.jpg")


# Sačuvaj kao sliku u latex formatu png
img3 = full_circuit.draw('latex')
img3.save("/data/qc30_5_3.png")


# Sačuvaj kao sliku u matplotlib formatu jpg
img4 = full_circuit.draw('mpl', fold=40)
img4.savefig("/data/qc30_5_4.jpg")

# Sačuvaj kao sliku u matplotlib formatu png
img5 = full_circuit.draw('mpl', fold=40)
img5.savefig("/data/qc30_5_5.png")
"""




# Sačuvaj kao sliku u matplotlib formatu jpg
img4 = full_circuit.draw('mpl', fold=40)
img4.savefig("/KvantniRegresor/1VQR/VQR_qc25_5_4.jpg")





###############################################







import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from qiskit_aer import AerSimulator
from qiskit_machine_learning.algorithms import NeuralNetworkRegressor
from qiskit_machine_learning.utils import algorithm_globals
from qiskit_machine_learning.optimizers import COBYLA
from tqdm import tqdm
import random

from qiskit.primitives import StatevectorSampler as Sampler
from qiskit_machine_learning.neural_networks import SamplerQNN, EstimatorQNN

from qiskit_machine_learning.optimizers import GradientDescent

from qiskit_aer.primitives import Sampler as AerSampler

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

import matplotlib.pyplot as plt

from qiskit_machine_learning.algorithms import VQR



# =========================
# 2. Koristimo svih 89 izvlacenja
# =========================
N = 89
df = df.tail(N).reset_index(drop=True)




X = df.iloc[:, :-1].values  # prvih 5 brojeva
y_full = df.values          # svi 6 brojeva (5+1)

# Skaliranje
scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X).astype(np.float64)

# =========================
# Treniranje i predikcija po brojevima
# =========================

predicted_combination = []
print()
for i in range(6):  # 5 brojeva + dodatni broj
    print(f"\n--- Treniranje QNN regresora za broj {i+1} ---")
    y = y_full[:, i].astype(np.float64)
    scaler_y = MinMaxScaler()
    y_scaled = scaler_y.fit_transform(y.reshape(-1,1)).ravel()

    # SamplerQNN sa lokalnim AerSimulator-om
    backend = AerSimulator()

    num_qubits = X_scaled.shape[1]
    print()
    print("\nnum_qubits")
    print(num_qubits, "\n")
    print()
    """
    
    """



    sampler = Sampler()

    
    gradient = GradientDescent()  # param-shift rule




    # -------------------------
    # Feature map sa num_qubits parametara
    # -------------------------
    param_x = ParameterVector("x", num_qubits)
    feature_map = QuantumCircuit(num_qubits, name="fm")

    for j in range(num_qubits):
        feature_map.ry(param_x[j], j)

    feature_map.barrier()
    for j in range(num_qubits - 1):
        feature_map.cz(j, j+1)
    feature_map.cz(num_qubits-1, 0)

    feature_map.draw("mpl", style="clifford")
    # plt.show()

    feature_map.decompose().draw(output="mpl", style="clifford", fold=20)
    # plt.show()

    # -------------------------
    # Ansatz sa num_qubits parametara
    # -------------------------
    param_y = ParameterVector("y", num_qubits)
    ansatz = QuantumCircuit(num_qubits, name="vf")

    for j in range(num_qubits):
        ansatz.ry(param_y[j], j)

    ansatz.barrier()
    for j in range(num_qubits - 1):
        ansatz.cz(j, j+1)
    ansatz.cz(num_qubits-1, 0)

    ansatz.draw("mpl", style="clifford")
    # plt.show()

    ansatz.decompose().draw(output="mpl", style="clifford", fold=20)
    # plt.show()








    # 3. Spoji ih u jedan parametarski krug
    full_circuit_map = feature_map.compose(ansatz)
    # full_circuit = feature_map.compose(feature_map)


    full_circuit.draw("mpl", style="clifford", fold=20)
    # plt.show()

    
    """
    # -------------------------
    # QNN (sada eksplicitno prosleđujemo parametre)
    # -------------------------
    regression_estimator_qnn = EstimatorQNN(
        circuit=full_circuit_map,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        gradient=gradient
    )
    """


    
    """
    def parity(x):
        return f"{bin(x)}".count("1") % 2


    regression_sampler_qnn = SamplerQNN(
        sampler=sampler,
        circuit=full_circuit_map,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        interpret=parity,
        output_shape=2,
        gradient=gradient
    )
    """
    

    

    # NeuralNetworkRegressor
    # optimizer = COBYLA(maxiter=1000)
    optimizer = COBYLA(maxiter=len(X_scaled))
    

    total_iters = len(X_scaled)
    pbar = tqdm(total=total_iters, desc=f"Broj {i+1}")

    def progress_callback(weights, loss):
        pbar.update(1)



    vqr = VQR(
        feature_map=feature_map,
        ansatz=ansatz,
        optimizer=optimizer,
        callback=progress_callback
    )




    
    # create empty array for callback to store evaluations of the objective function
    objective_func_vals = []
    plt.rcParams["figure.figsize"] = (12, 6)

    # Fit sa progres bar
    vqr.fit(X_scaled, y_scaled)
    pbar.close()

    # return to default figsize
    plt.rcParams["figure.figsize"] = (6, 4)

    

    # score result
    print("\vqr.score(X_scaled, y_scaled)")
    print(vqr.score(X_scaled, y_scaled), "\n")
    """
    qr.score(X_scaled, y_scaled)
    0.8616721019884289 
    """
    
    print("vqr.weights")
    print(vqr.weights, "\n")
    """
    vqr.weights
    [ 1.6261398  -0.02813501  2.50853307 -0.53617058 -0.01783437]
    """




    # plot data
    plt.plot(X_scaled, y_scaled, "bo")
    plt.title(f"Broj {i+1} - Podaci")
    plt.xlabel("Ulazni podaci (prvih 5 brojeva)")
    plt.ylabel(f"Izlazni podaci (broj {i+1})")
    plt.grid()
    # plt.show()




    # Predikcija sledećeg broja
    last_scaled = scaler_X.transform([X[-1]]).astype(np.float64)
    pred_scaled = vqr.predict(last_scaled)
    pred = scaler_y.inverse_transform(pred_scaled.reshape(-1,1))[0][0]
    pred = max(1, int(round(pred)))  # garantuje da nije 0

    predicted_combination.append(pred)
    print(f"Predikcija za broj {i+1}: {pred}")

print()
print("\n=== Predviđena sledeća loto kombinacija (5+1) ===")
print(" ".join(str(num) for num in predicted_combination))
print()
"""
89
89
=== Predviđena sledeća loto kombinacija (5+1) ===
4 6 x x 29 6
"""











"""
cisti kesh

pip cache purge

"""


"""
Obriši cache
Na Mac/Linux:
rm -rf ~/.cache/pip
"""


"""
=== Qiskit Version Table ===
Software                       Version        
---------------------------------------------
qiskit                         1.4.4          
qiskit_machine_learning        0.8.3          

=== System Information ===
Python version                 3.11.13        
OS                             Darwin         
Time                           Tue Sep 09 18:11:49 2025 CEST
"""



