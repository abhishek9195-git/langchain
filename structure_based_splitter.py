from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

code_text = """
class Car:

    # Class-level attribute (shared by all instances)
    is_a_vehicle = True

    def __init__(self, make, model, year, color):

        self.make = make
        self.model = model
        self.year = year
        self.color = color
        # Initial internal state properties
        self._mileage = 0
        self._engine_running = False

    # --- Methods (Actions the object can perform) ---

    def start_engine(self):
        if self._engine_running:
            return f"The {self.make} {self.model}'s engine is already running."
        else:
            self._engine_running = True
            return f"The {self.make} {self.model}'s engine started."

    def stop_engine(self):
        if not self._engine_running:
            return f"The {self.make} {self.model}'s engine is already off."
        else:
            self._engine_running = False
            return f"The {self.make} {self.model}'s engine stopped."
    
    def drive(self, miles):
        if self._engine_running:
            self._mileage += miles
            return f"Drove {miles} miles. Total mileage is now {self._mileage}."
        else:
            return "Cannot drive. Start the engine first!"

    # --- Properties (Managed attributes with getter/setter logic) ---

    @property
    def mileage(self):
        return self._mileage

    @property
    def full_name(self):
        return f"{self.year} {self.color} {self.make} {self.model}"

    @property
    def is_running(self):
        return self._engine_running
    
    @color.setter
    def color(self, new_color):
        if not isinstance(new_color, str) or len(new_color) < 2:
            raise ValueError("Color must be a valid string.")
        self._color = new_color


# --- Usage Example ---

# 1. Create an instance of the Car class
my_car = Car(make='Toyota', model='Camry', year=2023, color='Blue')

print(f"I own a {my_car.full_name}") # Accessing a property
print(f"Is this a vehicle? {my_car.is_a_vehicle}") # Accessing a class attribute

# 2. Call methods to interact with the car
print(my_car.start_engine())
print(my_car.drive(50))
print(my_car.drive(10))
print(my_car.stop_engine())

# 3. Accessing properties after interaction
print(f"Total Mileage: {my_car.mileage} miles")

# 4. Using the setter property
my_car.color = "Red"
print(f"My car is now: {my_car.color}")

"""

code_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=200,
    chunk_overlap=0
)

chunks = code_splitter.split_text(code_text)

print(f'===> Number of chunks {len(chunks)}')
print(chunks[0])
