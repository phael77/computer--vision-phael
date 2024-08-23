import os
from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Carregar o modelo YOLO
model = YOLO('yolov8n.pt')

# Diretório com as imagens de entrada
input_dir = 'c:\\Users\\rapha\\Desktop\\projetin\\imagens'  # Substitua pelo caminho da sua pasta
# Diretório para salvar as imagens resultantes
output_dir = 'c:\\Users\\rapha\\Desktop\\projetin\\detectadas'  # Substitua pelo caminho da pasta de saída

# Verificar se o diretório de entrada existe
if not os.path.exists(input_dir):
    raise FileNotFoundError(f"O diretório de entrada especificado não foi encontrado: {input_dir}")

# Certificar que o diretório de saída exista
os.makedirs(output_dir, exist_ok=True)

# Lista para armazenar a contagem de pessoas detectadas em cada imagem
person_counts = []

# Iterar sobre todas as imagens no diretório de entrada
for img_file in os.listdir(input_dir):
    # Caminho completo da imagem
    img_path = os.path.join(input_dir, img_file)
    
    # Realizar a previsão na imagem
    results = model(img_path)
    
    # Inicializar a contagem de pessoas para esta imagem
    person_count = 0
    
    # Verificar se foram detectadas pessoas (classe 0)
    for result in results:
        # Filtrar as detecções para manter apenas pessoas
        person_detections = [det for det in result.boxes if det.cls == 0]  # Filtra as detecções de pessoas
        
        if person_detections:
            # Incrementar a contagem de pessoas
            person_count += len(person_detections)
            
            # Salvar a imagem resultante com as detecções originais do YOLO
            save_path = os.path.join(output_dir, img_file)
            result.save(filename=save_path)  # Salva a imagem com as detecções originais do YOLO
            
            # Carregar e exibir a imagem resultante
            image = mpimg.imread(save_path)  # Carrega a imagem salva
            plt.imshow(image)
            plt.axis('off')
            plt.show()
    
    # Adicionar a contagem de pessoas para esta imagem na lista
    person_counts.append((img_file, person_count))

# Gerar um gráfico com a contagem de pessoas detectadas em cada imagem
img_files, counts = zip(*person_counts)  # Separar nomes dos arquivos e contagens

plt.figure(figsize=(5, 5))
plt.bar(img_files, counts, color='blue')
plt.xlabel('Imagem')
plt.ylabel('Número de Pessoas Detectadas')
plt.title('Contagem de Pessoas Detectadas em Cada Imagem')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
