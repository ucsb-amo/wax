from waxa.data.data_vault import DataVault as DataVaultWaxa

class DataVault(DataVaultWaxa):
    def __init__(self):
        super().__init__()
        
        self.images = self.add_data_container('images')
        self.image_timestamps = self.add_data_container('image_timestamps')