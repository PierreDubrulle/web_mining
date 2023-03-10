### Challenge web mining
## Reconnaissance faciale
# Détection des visages
Prédictions :

- prénom
- âge
- genre
- humeur
## Reconnaissance gestuelle
Démarrer et éteindre l'enregistrement d'une vidéo avec les mains  
Prendre une photo en montrant ses 10 doigts  

## Reconnaissance de la voix
Allumage de la webcam  
Prise de photo  

### Utilisation des différents fonctionnalités

Faire un git clone du repository  
Se positionner dans le dossier  
Créer un environnement virtuel et l'activer  
Installer les librairies avec  

```python
pip install -r requirements.txt
```
Lancer les différentes fonctions
```python
python <nomdufichier.py>
```

### Docker

## WARNING : only for Linux

Se rendre dans le dossier docker.

```console
xhost +local:docker
```
build
```console
dokcer build -t web_mining .
```
run
```console
docker run -ti --rm --net=host -p 8501:8501 --name app --volume="$HOME/.Xauthority:/root/.Xauthority:rw" --env="DISPLAY" --device=/dev/video0:/dev/video0 --device /dev/snd:/dev/snd web_mining
```

L'application est visible sur le port 8501 en localhost, ou regarder l'url indiquée par le prompt.
