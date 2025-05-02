import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

class MarketAnalyzer:
    def __init__(self):
        # Sample art prices by style
        self.style_prices = {
            "Impressionism": (5000, 50000),
            "Expressionism": (3000, 40000),
            "Cubism": (8000, 60000),
            "Surrealism": (4000, 45000),
            "Abstract Art": (2000, 35000),
            "Realism": (6000, 55000),
            "Pop Art": (7000, 65000),
            "Contemporary Art": (1000, 30000),
            "Classical Art": (9000, 70000),
            "Digital Art": (500, 20000)
        }
        
        # Customer profiles by style
        self.style_customers = {
            "Impressionism": "Traditional collectors, art galleries, museums",
            "Expressionism": "Modern art collectors, contemporary galleries",
            "Cubism": "Modern art collectors, modern art museums",
            "Surrealism": "Surrealist art collectors, specialized galleries",
            "Abstract Art": "Contemporary collectors, interior designers",
            "Realism": "Traditional collectors, educational institutions",
            "Pop Art": "Pop art collectors, modern art galleries",
            "Contemporary Art": "Young collectors, emerging galleries",
            "Classical Art": "Traditional collectors, classical art museums",
            "Digital Art": "Digital art collectors, virtual galleries"
        }
        
    def get_price_range(self, style):
        return self.style_prices.get(style, (1000, 20000))
    
    def get_customer_profile(self, style):
        return self.style_customers.get(style, "General collectors")
    
    def get_marketing_strategy(self, style):
        strategies = {
            "Impressionism": """
            - Traditional gallery exhibitions
            - Participation in classical art fairs
            - Museum collaborations
            - Marketing in traditional art magazines
            """,
            "Expressionism": """
            - Modern art galleries
            - Contemporary art fairs
            - Art-focused social media
            - Collaborations with emerging artists
            """,
            "Digital Art": """
            - NFT platforms
            - Virtual galleries
            - Digital social media
            - Metaverse exhibitions
            """
        }
        return strategies.get(style, """
        - Local gallery exhibitions
        - Social media presence
        - Art fair participation
        - Artist collaborations
        """) 