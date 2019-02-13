# Specific classes subset
specific_classes = {'Alilaguna',
                    'Lanciafino10m',
                    'Lanciafino10mBianca',
                    'Lanciafino10mMarrone',
                    'Lanciamaggioredi10mBianca',
                    'MotoscafoACTV',
                    'VaporettoACTV',
                    'Motobarca',
                    'Mototopo',
                    'Motopontonerettangolare',
                    'Raccoltarifiuti',
                    'Barchino',
                    'Patanella',
                    'Topa',
                    'Gondola',
                    'Sandoloaremi',
                    'Polizia',
                    'Ambulanza',
                    'Water'
                    }


# general classes
def general_classes(label):
    switch = {
        'Alilaguna': people_transport,
        'Lanciafino10m': people_transport,
        'Lanciafino10mBianca': people_transport,
        'Lanciafino10mMarrone': people_transport,
        'Lanciamaggioredi10mBianca': people_transport,
        'Lanciamaggioredi10mMarrone': people_transport,
        'MotoscafoACTV': people_transport,
        'VaporettoACTV': people_transport,
        'Motobarca': general_transport,
        'Mototopo': general_transport,
        'Motopontonerettangolare': general_transport,
        'Raccoltarifiuti': general_transport,
        'Barchino': pleasure_craft,
        'Cacciapesca': pleasure_craft,
        'Patanella': pleasure_craft,
        'Sanpierota': pleasure_craft,
        'Topa': pleasure_craft,
        'Gondola': rowing_transport,
        'Caorlina': rowing_transport,
        'Sandoloaremi': rowing_transport,
        'Polizia': public_utility,
        'Ambulanza': public_utility,
        'VigilidelFuoco': public_utility,
        'Water': water
    }
    return switch[label]()


# define the function blocks
def people_transport():
    return 'People Transport'


def general_transport():
    return 'General Transport'


def pleasure_craft():
    return 'Pleasure Craft'


def rowing_transport():
    return 'Rowing Transport'


def public_utility():
    return 'Public Utility'


def water():
    return 'Water'
