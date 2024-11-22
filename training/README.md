# Running training

Locally, you can run:

- docker compose up --build
  which will run the api as well, where the model is lives.

Longer term we probably want to load the model from it's own module, in both places

Then

- docker exec -it mlx-caption-deployment-training-1 sh
- python -m training.training
  or
- python -m training.finetune

The latter will pull user-generated training data from the postgresdb
