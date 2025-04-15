import cv2
import numpy as np
import gradio as gr

src_points = []
def reset(out):
    src_points.clear()
    n_coordinates = 0
    out = np.zeros((512, 512, 3), dtype='uint8')
    return n_coordinates, out

def on_select(value, evt: gr.EventData):
    if len(src_points) < 4:
        src_points.append(evt._data['index'])
    return len(src_points)

def fixImg(inp):
    
    dst_points = np.float32((
        [0,0],
        [0,800],
        [600,800],
        [600,0]
    ))
    
    src_float = np.float32(src_points)
    H = cv2.getPerspectiveTransform(src_float, dst_points)
    output_img = cv2.warpPerspective(inp, H, (600,800))

    return output_img

# Interface
with gr.Blocks() as demo:
    # è possibile utilizzare il markdown per formattare il testo
    gr.Markdown('## Document Scanner')
    # Stampa variabili in un box 
    n_coordinates = gr.Textbox(label="Number of Coordinates", value=0)

    # Gli elementi possono essere organizzati in righe e colonnne
    with gr.Row():
        # Carica un'immagine
        inp = gr.Image(label="Input Image")
        out = gr.Image(label="Output Image")

        # Aggiunge un evento di selezione, in particolare quando si clicca su un punto dell'immagine
        inp.select(fn=on_select, inputs=[inp], outputs=[n_coordinates])
    
    with gr.Row():
        # Aggiunge un pulsante per eseguire la funzione fixImg
        btn = gr.Button("Scan Document")
        btn.click(fn=fixImg, inputs=[inp], outputs=[out])

        # Aggiunge un pulsante per resettare i punti selezionati
        btn = gr.Button("Reset")
        btn.click(fn=reset,inputs=[out], outputs=[n_coordinates, out])
    

demo.launch()