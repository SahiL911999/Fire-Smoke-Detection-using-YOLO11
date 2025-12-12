"""
Pascal VOC to YOLO Format Converter
=====================================

This script converts object detection annotations from Pascal VOC XML format to YOLO format.

Pascal VOC Format (XML):
- XML file with bounding boxes as absolute coordinates: <xmin>, <ymin>, <xmax>, <ymax>
- One XML file per image

YOLO Format (TXT):
- Text file with normalized coordinates: <class_id> <x_center> <y_center> <width> <height>
- Values normalized to [0, 1] range
- One TXT file per image

Author: YOLO WildFire Detection Project
Date: 2025
"""

import os
import xml.etree.ElementTree as ET
from pathlib import Path
import shutil
from tqdm import tqdm


class VOCtoYOLOConverter:
    """
    Converts Pascal VOC format annotations to YOLO format.
    """
    
    def __init__(self, class_mapping=None):
        """
        Initialize the converter.
        
        Args:
            class_mapping (dict): Dictionary mapping VOC class names to YOLO class IDs
                                Example: {'smoke': 0, 'fire': 1}
        """
        self.class_mapping = class_mapping or {'smoke': 0, 'fire': 1}
        
    def convert_box_coordinates(self, size, box):
        """
        Convert Pascal VOC box coordinates to YOLO format.
        
        Args:
            size (tuple): Image dimensions (width, height)
            box (tuple): VOC bounding box (xmin, ymin, xmax, ymax)
            
        Returns:
            tuple: YOLO format (x_center, y_center, width, height) normalized to [0, 1]
        """
        dw = 1.0 / size[0]  # Normalize by image width
        dh = 1.0 / size[1]  # Normalize by image height
        
        # Calculate center point and dimensions
        x_center = (box[0] + box[2]) / 2.0
        y_center = (box[1] + box[3]) / 2.0
        width = box[2] - box[0]
        height = box[3] - box[1]
        
        # Normalize to [0, 1]
        x_center = x_center * dw
        y_center = y_center * dh
        width = width * dw
        height = height * dh
        
        return (x_center, y_center, width, height)
    
    def parse_voc_xml(self, xml_file):
        """
        Parse a VOC XML annotation file.
        
        Args:
            xml_file (str): Path to VOC XML file
            
        Returns:
            tuple: (image_size, annotations_list)
                - image_size: (width, height)
                - annotations_list: List of (class_id, bbox) tuples
        """
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        # Get image dimensions
        size = root.find('size')
        img_width = int(size.find('width').text)
        img_height = int(size.find('height').text)
        
        annotations = []
        
        # Parse all objects in the image
        for obj in root.iter('object'):
            class_name = obj.find('name').text
            
            # Skip if class not in mapping
            if class_name not in self.class_mapping:
                print(f"Warning: Class '{class_name}' not in class mapping. Skipping.")
                continue
            
            class_id = self.class_mapping[class_name]
            
            # Get bounding box coordinates
            xmlbox = obj.find('bndbox')
            xmin = float(xmlbox.find('xmin').text)
            ymin = float(xmlbox.find('ymin').text)
            xmax = float(xmlbox.find('xmax').text)
            ymax = float(xmlbox.find('ymax').text)
            
            # Convert to YOLO format
            bbox = self.convert_box_coordinates(
                (img_width, img_height), 
                (xmin, ymin, xmax, ymax)
            )
            
            annotations.append((class_id, bbox))
        
        return (img_width, img_height), annotations
    
    def convert_file(self, xml_path, output_path):
        """
        Convert a single VOC XML file to YOLO format.
        
        Args:
            xml_path (str): Path to input VOC XML file
            output_path (str): Path to output YOLO TXT file
        """
        try:
            img_size, annotations = self.parse_voc_xml(xml_path)
            
            # Write YOLO format file
            with open(output_path, 'w') as f:
                for class_id, bbox in annotations:
                    # YOLO format: <class_id> <x_center> <y_center> <width> <height>
                    f.write(f"{class_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")
            
            return True
        except Exception as e:
            print(f"Error converting {xml_path}: {str(e)}")
            return False
    
    def convert_dataset(self, voc_annotations_dir, images_dir, output_dir, copy_images=True):
        """
        Convert an entire dataset from VOC to YOLO format.
        
        Args:
            voc_annotations_dir (str): Directory containing VOC XML files
            images_dir (str): Directory containing corresponding images
            output_dir (str): Output directory for YOLO format dataset
            copy_images (bool): Whether to copy images to output directory
        """
        # Create output directories
        output_labels_dir = os.path.join(output_dir, 'labels')
        output_images_dir = os.path.join(output_dir, 'images')
        
        os.makedirs(output_labels_dir, exist_ok=True)
        if copy_images:
            os.makedirs(output_images_dir, exist_ok=True)
        
        # Get all XML files
        xml_files = list(Path(voc_annotations_dir).glob('*.xml'))
        
        print(f"Found {len(xml_files)} XML annotation files")
        print(f"Converting to YOLO format...")
        
        successful = 0
        failed = 0
        
        for xml_file in tqdm(xml_files, desc="Converting annotations"):
            # Get base filename without extension
            base_name = xml_file.stem
            
            # Output label file path
            output_label = os.path.join(output_labels_dir, f"{base_name}.txt")
            
            # Convert annotation
            if self.convert_file(str(xml_file), output_label):
                successful += 1
                
                # Copy corresponding image if requested
                if copy_images:
                    # Try different image extensions
                    for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                        img_path = os.path.join(images_dir, f"{base_name}{ext}")
                        if os.path.exists(img_path):
                            output_img = os.path.join(output_images_dir, f"{base_name}{ext}")
                            shutil.copy2(img_path, output_img)
                            break
            else:
                failed += 1
        
        print(f"\n✅ Conversion Complete!")
        print(f"Successfully converted: {successful} files")
        print(f"Failed: {failed} files")
        print(f"Output directory: {output_dir}")


def main():
    """
    Example usage of the VOC to YOLO converter.
    """
    # Define class mapping (customize based on your dataset)
    class_mapping = {
        'smoke': 0,
        'fire': 1
    }
    
    # Initialize converter
    converter = VOCtoYOLOConverter(class_mapping=class_mapping)
    
    # Example: Convert a single file
    # converter.convert_file('path/to/annotation.xml', 'path/to/output.txt')
    
    # Example: Convert entire dataset
    """
    converter.convert_dataset(
        voc_annotations_dir='path/to/voc/annotations',
        images_dir='path/to/voc/images',
        output_dir='path/to/yolo/dataset',
        copy_images=True
    )
    """
    
    print("VOC to YOLO Converter initialized.")
    print("Modify the paths in main() function to convert your dataset.")
    print("\nClass Mapping:")
    for class_name, class_id in class_mapping.items():
        print(f"  {class_name} -> {class_id}")


if __name__ == "__main__":
    main()
