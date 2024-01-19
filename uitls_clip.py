
import cv2
import numpy as np
import torch 
from PIL import Image
from FastSAM.fastsam.utils import image_to_np_ndarray

try:
    import clip  # for linear_assignment
except (ImportError, AssertionError, AttributeError):
    from ultralytics.yolo.utils.checks import check_requirements
    check_requirements('git+https://github.com/openai/CLIP.git')  # required before installing lap from source
    import clip


class FastSAMPrompt:
    def __init__(self, image, results, device='cuda'):
        if isinstance(image, str) or isinstance(image, Image.Image):
            image = image_to_np_ndarray(image)
        self.device = device
        self.results = results
        self.img = image
        self.lc_list = [
            # bld related.
            'roof', 'rooftop', 'building', 'house', 'apartment', 'residential', 'factory', 
            # non-bld related.
            'vegetation', 'tree', 'vehicle', 'playground', 'baseball diamond', 'swimming pool', 'roundabout', 'basketball court', 'bareland', 'sand'
        ]

    # clip
    @torch.no_grad()
    def retrieve(self, model, preprocess, elements, device) -> int:
        # single img.
        # preprocessed_images = preprocess(elements[0]).unsqueeze(0).to(device)
        # multiple img.
        preprocessed_images = [preprocess(image).to(device) for image in elements]
        stacked_images = torch.stack(preprocessed_images)

        tokenized_text = torch.cat([clip.tokenize(f"satellite image of {c}") for c in self.lc_list]).to(device)

        image_features = model.encode_image(stacked_images)
        text_features = model.encode_text(tokenized_text)
        
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        # probs = 100.0 * image_features @ text_features.T
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        return similarity.cpu().numpy()
    


    def _segment_image(self, image, bbox):
        if isinstance(image, Image.Image):
            image_array = np.array(image)
        else:
            image_array = image
        segmented_image_array = np.zeros_like(image_array)
        x1, y1, x2, y2 = bbox
        segmented_image_array[y1:y2, x1:x2] = image_array[y1:y2, x1:x2]
        segmented_image = Image.fromarray(segmented_image_array)
        black_image = Image.new('RGB', image.size, (255, 255, 255))
        # transparency_mask = np.zeros_like((), dtype=np.uint8)
        transparency_mask = np.zeros((image_array.shape[0], image_array.shape[1]), dtype=np.uint8)
        transparency_mask[y1:y2, x1:x2] = 255
        transparency_mask_image = Image.fromarray(transparency_mask, mode='L')
        black_image.paste(segmented_image, mask=transparency_mask_image)
        return black_image
    
    def _get_bbox_from_mask(self, mask):
        mask = mask.astype(np.uint8)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        x1, y1, w, h = cv2.boundingRect(contours[0])
        x2, y2 = x1 + w, y1 + h
        if len(contours) > 1:
            for b in contours:
                x_t, y_t, w_t, h_t = cv2.boundingRect(b)
                # Merge multiple bounding boxes into one.
                x1 = min(x1, x_t)
                y1 = min(y1, y_t)
                x2 = max(x2, x_t + w_t)
                y2 = max(y2, y_t + h_t)
            h = y2 - y1
            w = x2 - x1
        return [x1, y1, x2, y2]

    def _crop_image(self, format_results):

        image = Image.fromarray(cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB))
        ori_w, ori_h = image.size
        annotations = format_results
        mask_h, mask_w = annotations[0]['segmentation'].shape
        if ori_w != mask_w or ori_h != mask_h:
            image = image.resize((mask_w, mask_h))
        cropped_boxes = []
        cropped_images = []
        select_mask = []
        filter_id = []
        # annotations, _ = filter_masks(annotations)
        # filter_id = list(_)
        for _, mask in enumerate(annotations):
            if np.sum(mask['segmentation']) <= 100:
                filter_id.append(_)
                continue
            bbox = self._get_bbox_from_mask(mask['segmentation'])  # mask 的 bbox
            cropped_boxes.append(self._segment_image(image, bbox))  
            # cropped_boxes.append(segment_image(image,mask["segmentation"]))
            cropped_images.append(bbox)  # Save the bounding box of the cropped image.
            select_mask.append(mask['segmentation']) # Save mask

        return cropped_boxes, cropped_images, select_mask, filter_id, annotations
    
    
    def _format_results(self, result, filter=0):
        annotations = []
        n = len(result.masks.data)
        for i in range(n):
            annotation = {}
            # print(result)
            mask = result.masks.data[i] == 1.0

            if torch.sum(mask) < filter:
                continue
            annotation['id'] = i
            annotation['segmentation'] = mask.cpu().numpy()
            annotation['bbox'] = result.boxes.data[i]
            annotation['score'] = result.boxes.conf[i]
            annotation['area'] = annotation['segmentation'].sum()
            annotations.append(annotation)
        return annotations
    

    def text_prompt(self, clip_model, preprocess):
        if self.results == None:
            return []
        format_results = self._format_results(self.results[0], 0)
        # print(format_results)
        cropped_boxes, cropped_images, select_mask, filter_id, annotations = self._crop_image(format_results)

        similarity = self.retrieve(clip_model, preprocess, cropped_boxes, device=self.device)


        mask_h, mask_w = annotations[0]['segmentation'].shape
        build_cons_scores = np.zeros((mask_h, mask_w, similarity.shape[0]))

        assert similarity.shape[0] == len(select_mask)

        for n in range(len(select_mask)):
            prob = similarity[n]
            bld_relate_prob = np.sum(prob[:6])
            
            build_cons_scores[:,:,n] = select_mask[n].astype(np.float32) * bld_relate_prob

        max_bld_score_map = np.max(build_cons_scores, axis=2).astype(np.float32)
        # max_bld_score_map = (max_bld_score_map * 255).astype(np.uint8)
        return max_bld_score_map
