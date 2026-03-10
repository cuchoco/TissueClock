from .pan import train as train_pan
from .tissue import train as train_tissue
from .abmil import train as train_abmil

TRAINERS = {
    'pan': train_pan,
    'tissue': train_tissue,
    'abmil': train_abmil,
}
