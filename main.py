import cv2 as cv
from ultralytics import YOLO
model = YOLO('yolov8n.pt')

input_video = cv.VideoCapture('./assets/arsene.mp4') # Abre o video de entrada

if not input_video.isOpened():
    print("Error opening video file")
    exit(1)

# Como foi possível abrir o video de entrada, vamos agora utilizar 
# essa captura para definir o tamanho do video de saida
width  = int(input_video.get(cv.CAP_PROP_FRAME_WIDTH))   # float `width`
height = int(input_video.get(cv.CAP_PROP_FRAME_HEIGHT))

# Cria a estrutura do video de saida
output_video = cv.VideoWriter( './output/out.avi',cv.VideoWriter_fourcc(*'DIVX'), 24, (width, height))

while True:
    isTrue, frame = input_video.read() 
    
    results = model(frame) # Joga o frame no modelo pré treinado

    resulted_frame = results[0].plot() # Extray o frame do array

    if not isTrue: # Se não tiver mais frames, encerra o playback
        break
    
    # Escreve o frame no output se tiverem pessoas na imagem
    person = False # Variável que recebe se tem pessoas na imagem
    for result in results: # Para cada resultado
            for box in result.boxes.cpu().numpy(): # Para cada box na imagem
                cls = int(box.cls[0]) # Classe da box
                if cls == 0: # Se a classe for 0, é uma pessoa
                    person = True
                    break
    if person: # Se tiver pessoas na imagem, escreve o frame com detecção no output e o apresenta
        cv.imshow('Video Playback', resulted_frame)
        output_video.write(resulted_frame)
    else: # Se não tiver pessoas na imagem, apenas mostra o frame padrão
        cv.imshow('Video Playback', frame)

    # Se o usuario apertar d, encerra o playback
    # O valor utilizado no waiKey define o fps do playback
    if cv.waitKey(20) & 0xFF == ord('d'):
        break
    
# Fecha tudo
output_video.release()
input_video.release()
cv.destroyAllWindows()