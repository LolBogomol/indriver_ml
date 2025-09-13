import torch
import numpy as np
from PIL import Image
from transformers import (
    DetrImageProcessor,
    DetrForObjectDetection,
    SamModel,
    SamProcessor,
    AutoImageProcessor,
)
import math


class CarExtractor:
    def __init__(
            self,
            sam_model_id: str = "facebook/sam-vit-base",
            detection_threshold: float = 0.7,
            downstream_model_id: str = "google/vit-base-patch16-224-in21k",
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Используемое устройство: {self.device}")

        detr_model_id = "facebook/detr-resnet-50"
        self.detr_processor = DetrImageProcessor.from_pretrained(detr_model_id)
        self.detr_model = DetrForObjectDetection.from_pretrained(detr_model_id).to(self.device)

        self.sam_model = SamModel.from_pretrained(sam_model_id).to(self.device)
        self.sam_processor = SamProcessor.from_pretrained(sam_model_id)
        self.detection_threshold = detection_threshold

        print(f"Загрузка процессора для следующей модели: {downstream_model_id}")
        self.downstream_processor = AutoImageProcessor.from_pretrained(downstream_model_id)

    def _detect_cars(self, image: Image.Image) -> list:
        inputs = self.detr_processor(images=image, return_tensors="pt").to(self.device)
        outputs = self.detr_model(**inputs)
        target_sizes = torch.tensor([image.size[::-1]]).to(self.device)
        results = self.detr_processor.post_process_object_detection(
            outputs,
            target_sizes=target_sizes,
            threshold=self.detection_threshold
        )[0]

        detected_cars = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            if self.detr_model.config.id2label[label.item()] == "car":
                detected_cars.append({
                    'score': score.item(),
                    'box': [b.item() for b in box]
                })
        return detected_cars

    def _find_main_car(self, image: Image.Image, detected_cars: list) -> dict | None:
        if not detected_cars:
            return None
        image_center = (image.width / 2, image.height / 2)

        for car in detected_cars:
            x0, y0, x1, y1 = car['box']
            car['area'] = (x1 - x0) * (y1 - y0)
            box_center_x = (x0 + x1) / 2
            box_center_y = (y0 + y1) / 2
            distance = math.sqrt((box_center_x - image_center[0]) ** 2 + (box_center_y - image_center[1]) ** 2)
            car['centrality'] = 1.0 / (distance + 1.0)

        max_area = max(car['area'] for car in detected_cars) or 1
        max_centrality = max(car['centrality'] for car in detected_cars) or 1
        max_score = max(car['score'] for car in detected_cars) or 1

        for car in detected_cars:
            car['norm_area'] = car['area'] / max_area
            car['norm_centrality'] = car['centrality'] / max_centrality
            car['norm_score'] = car['score'] / max_score

        best_car = max(detected_cars, key=lambda car:
        0.5 * car['norm_area'] +
        0.3 * car['norm_centrality'] +
        0.2 * car['norm_score'])

        return best_car

    def _segment_boxes(self, image: Image.Image, boxes: list) -> torch.Tensor:
        all_masks = []
        for box in boxes:
            # --- ИСПРАВЛЕНИЕ ЗДЕСЬ ---
            # Добавляем ещё один уровень вложенности для соответствия формату
            inputs = self.sam_processor(image, input_boxes=[[box]], return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.sam_model(**inputs)

            mask = self.sam_processor.image_processor.post_process_masks(
                outputs.pred_masks.cpu(),
                inputs["original_sizes"].cpu(),
                inputs["reshaped_input_sizes"].cpu()
            )[0]

            all_masks.append(mask)
        if not all_masks: return torch.empty(0)
        return torch.cat(all_masks, dim=0)

    def _crop_with_mask(self, image: Image.Image, mask: torch.Tensor, box: list) -> Image.Image:
        mask_np = mask[0].cpu().numpy().astype(bool)
        image_rgba_np = np.array(image.convert("RGBA"))
        masked_image_np = image_rgba_np.copy()
        masked_image_np[~mask_np, 3] = 0
        x0, y0, x1, y1 = [int(coord) for coord in box]
        cropped_image_np = masked_image_np[y0:y1, x0:x1]
        return Image.fromarray(cropped_image_np)

    def extract_cars(self, image: Image.Image) -> list[Image.Image]:
        print("\n1. Поиск всех машин-кандидатов...")
        image_rgb = image.convert("RGB")
        all_detected_cars = self._detect_cars(image_rgb)

        if not all_detected_cars:
            print("Машины не найдены.")
            return []

        print(f"Найдено машин-кандидатов: {len(all_detected_cars)}. Выбираю главную...")
        main_car = self._find_main_car(image, all_detected_cars)

        if not main_car:
            print("Не удалось выбрать главную машину.")
            return []

        main_car_box_list = [main_car['box']]

        print("Сегментирую главную машину...")
        all_masks = self._segment_boxes(image_rgb, main_car_box_list)

        print("Вырезаю объект...")
        mask = all_masks[0]
        car_image = self._crop_with_mask(image, mask, main_car['box'])

        print("Готово!")
        return [car_image]

    def save_for_test(self, image: Image.Image, output_path: str):
        print(f"  -> Сохранение тестового изображения в: {output_path}")
        image.save(output_path, "PNG")

    def get_tensors(self, car_images: list[Image.Image]) -> torch.Tensor:
        if not car_images: return torch.empty(0)
        print(f"\n2. Подготовка {len(car_images)} изображений в виде тензоров...")
        rgb_car_images = [img.convert("RGB") for img in car_images]
        inputs = self.downstream_processor(
            images=rgb_car_images,
            return_tensors="pt"
        )
        return inputs.pixel_values.to(self.device)


if __name__ == "__main__":
    extractor = CarExtractor()
    try:
        source_image = Image.open("car.png")
    except FileNotFoundError:
        print("Ошибка: файл не найден.")
        exit()

    list_of_main_car = extractor.extract_cars(source_image)

    if list_of_main_car:
        main_car_image = list_of_main_car[0]
        extractor.save_for_test(main_car_image, "main_car_extracted.png")
        tensors_batch = extractor.get_tensors([main_car_image])

        print("\n--- Результат ---")
        print(f"Готовый батч тензоров имеет форму: {tensors_batch.shape}")
