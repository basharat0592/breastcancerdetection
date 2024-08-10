import gradio as gr
import requests

def predict(mean_radius, mean_texture):
    # Replace with your FastAPI endpoint URL
    response = requests.post('http://localhost:8000/predict', json={
        'mean_radius': mean_radius,
        'mean_texture': mean_texture
    })
    return response.json()['prediction']

# Create Gradio interface
iface = gr.Interface(
    fn=predict,
    inputs=[gr.inputs.Number(label='Mean Radius'), gr.inputs.Number(label='Mean Texture')],
    outputs=gr.outputs.Textbox(label='Prediction')
)

if __name__ == '__main__':
    iface.launch()
