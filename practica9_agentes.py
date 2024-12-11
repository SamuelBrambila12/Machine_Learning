import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Constantes
PEOPLE = 50
INCOME_MAX = 30
MAX_STEPS = 200  # Limitar los pasos a 200

# Variables globales
bank_loans = 0
bank_reserves = 0
bank_deposits = 0
bank_to_loan = 0
rich = 0
poor = 0
middle_class = 0

# Definir la clase de los agentes
class Turtle:
    def __init__(self):
        self.wallet = random.randint(0, 2 * INCOME_MAX)
        self.savings = 0
        self.loans = 0
        self.wealth = 0
        self.customer = None
        self.color = 'gray'
        self.update_color()
        
        # Posición en la cuadrícula
        self.x = random.uniform(-1, 1)
        self.y = random.uniform(-1, 1)

    def update_color(self):
        if self.savings > INCOME_MAX:
            self.color = 'green'  # Verde para los ricos
        elif self.loans > INCOME_MAX:
            self.color = 'red'    # Rojo para los pobres
        else:
            self.color = 'gray'   # Gris para la clase media
        self.wealth = self.savings - self.loans

    def balance_books(self):
        global bank_to_loan
        if self.wallet < 0:
            if self.savings >= -self.wallet:
                self.withdraw_from_savings(-self.wallet)
            else:
                self.withdraw_from_savings(self.savings)
                temp_loan = bank_to_loan
                if temp_loan >= -self.wallet:
                    self.take_out_a_loan(-self.wallet)
                else:
                    self.take_out_a_loan(temp_loan)
        else:
            self.deposit_to_savings(self.wallet)

        # Repagar préstamos si hay suficientes ahorros
        if self.loans > 0 and self.savings > 0:
            if self.savings >= self.loans:
                self.repay_a_loan(self.loans)
            else:
                self.repay_a_loan(self.savings)

    def do_business(self, other):
        if ((self.savings > 0 or self.wallet > 0 or bank_to_loan > 0) and
            random.random() < 0.5):  # 50% de probabilidad de comercio
            amount = 5 if random.random() < 0.5 else 2
            self.wallet -= amount
            other.wallet += amount

    def deposit_to_savings(self, amount):
        self.wallet -= amount
        self.savings += amount

    def withdraw_from_savings(self, amount):
        self.wallet += amount
        self.savings -= amount

    def repay_a_loan(self, amount):
        global bank_to_loan
        self.loans -= amount
        self.wallet -= amount
        bank_to_loan += amount

    def take_out_a_loan(self, amount):
        global bank_to_loan
        self.loans += amount
        self.wallet += amount
        bank_to_loan -= amount

# Inicializar los turtles
turtles = [Turtle() for _ in range(PEOPLE)]

# Funciones de manejo bancario
def setup_bank():
    global bank_loans, bank_reserves, bank_deposits, bank_to_loan
    bank_loans = sum(t.loans for t in turtles)
    bank_deposits = sum(t.savings for t in turtles)
    bank_reserves = 0.1 * bank_deposits  # Ejemplo de una reserva del 10%
    bank_to_loan = bank_deposits - (bank_reserves + bank_loans)

def poll_class():
    global rich, poor, middle_class
    rich = sum(1 for t in turtles if t.savings > INCOME_MAX)
    poor = sum(1 for t in turtles if t.loans > INCOME_MAX)
    middle_class = PEOPLE - (rich + poor)

# Animación
fig, ax = plt.subplots()
sc = ax.scatter([t.x for t in turtles], [t.y for t in turtles], c=[t.color for t in turtles])

# Configuración de título y leyenda(simbología)
ax.set_title('Simulación Económica y Bancaria')
rich_patch = plt.Line2D([0], [0], marker='o', color='w', label='Ricos', markerfacecolor='green', markersize=10)
middle_class_patch = plt.Line2D([0], [0], marker='o', color='w', label='Clase Media', markerfacecolor='gray', markersize=10)
poor_patch = plt.Line2D([0], [0], marker='o', color='w', label='Pobres', markerfacecolor='red', markersize=10)
ax.legend(handles=[rich_patch, middle_class_patch, poor_patch], loc='upper right')

# Título dinámico
title = ax.set_title("Simulación Económica y Bancaria - Paso 0")

# Texto para mostrar el conteo de clases
class_count_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, fontsize=12, va='top', ha='left', bbox=dict(facecolor='white', alpha=0.7))

def update(frame):
    if frame >= MAX_STEPS:
        ani.event_source.stop()  # Detener la animación después de 100 pasos
        return

    for t in turtles:
        other = random.choice(turtles)
        if other != t:
            t.do_business(other)
        t.balance_books()
        t.update_color()

    setup_bank()
    poll_class()
    
    # Actualizar dispersión de colores y posiciones
    sc.set_offsets([[t.x, t.y] for t in turtles])
    sc.set_color([t.color for t in turtles])

    # Actualizar el título con el paso actual
    title.set_text(f"Simulación Económica y Bancaria - Paso {frame+1}")

    # Actualizar el conteo de clases en el texto
    class_count_text.set_text(f"Ricos: {rich}\nClase Media: {middle_class}\nPobres: {poor}")

ani = animation.FuncAnimation(fig, update, frames=MAX_STEPS + 1, interval=200, blit=False)
plt.show()
