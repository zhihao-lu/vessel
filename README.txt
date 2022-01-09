Machine learning model using YOLOv3 to detect ships and kayaks along the coast of Singapore islands. Project for DSA4261 in conjunction with GovTech

Folder structure:

<root_folder>
├───best.pt (or any other .pt file)
├───main.py
├───requirements.txt 
├───Dockerfile
├───*.json (input file)
├───*.avi(input file)

Arguments:
--vid : path to .avi file
--json: path to .json file
--weights (optional): path to weights file if file is not named 'best.pt'

Output:
Output will saved in the volume. 

Docker image:
When building the docker image, we have to create a volume to allow for input and output. When doing docker run, it takes in 3 arguments,
2 of them required, 1 optional.

Example docker commands:
docker build .
docker run --rm -v <absolute/path/to/root_folder>/:/app <IMAGE_ID> --vid horizon_1_ship.avi --json horizon_1_ship.json
