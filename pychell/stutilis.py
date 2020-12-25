import streamlit as st


class StreamlitComponent:
    
    def __init__(self, label="", *args, **kwargs):
        self.label = label
        
    def write(self, *args, **kwargs):
        pass