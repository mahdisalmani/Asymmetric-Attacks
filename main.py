import src.models.utils as model_utils
from src.models.model import ModelWrapper
from utils.data import open_image, open_labels, get_label_by_name, check_image, get_mean_std
from utils.args import get_args, get_attack
from tqdm import tqdm
import pickle


args = get_args()
labels = open_labels(args.data_dir)
mean, std = get_mean_std(model_name=args.model, device=args.device)
model = ModelWrapper(model_utils.get_model(model_name=args.model).to(device=args.device), mean, std)
attack = get_attack(args=args, model=model)
logs = []

for i in tqdm(range(args.image_start, args.image_end), desc="Running The Attack"):
    image_name = f"{args.image_prefix}{'%08d' % i}.JPEG"
    image_path = f"{args.data_dir}/{image_name}"
    image_label = get_label_by_name(labels, image_name)
    image = model_utils.preprocess(open_image(source_image=image_path), mean=model.mean, std=model.std, model_name=args.model).to(args.device)
    if not check_image(source_image=image, model=model, true_label=image_label):
        continue
    perturbed_image, current_cost = attack(image, image_label)
    # print(attack.logs)
    logs.append([image_name, attack.logs])

    if args.save_logs:
        with open(f"attack_{args.attack}_{args.total_cost}_{args.query_cost}_{args.search_cost}_{args.overshooting}_{args.overshooting_scheduler_init}.log", "wb") as fp:
            pickle.dump(logs, fp)