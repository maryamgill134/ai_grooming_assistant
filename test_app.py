"""
Unit tests for AI Grooming Assistant
Run with: pytest tests/test_app.py -v
"""

import os
import sys
import json
from pathlib import Path
import unittest
from unittest.mock import patch, MagicMock

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

class TestAppInitialization(unittest.TestCase):
    """Test app initialization"""
    
    def test_imports(self):
        """Test if all modules can be imported"""
        try:
            from app import app
            from predict import predict_attributes
            from config import get_config, DevelopmentConfig
            self.assertIsNotNone(app)
            self.assertIsNotNone(predict_attributes)
            self.assertIsNotNone(DevelopmentConfig)
        except ImportError as e:
            self.fail(f"Failed to import required modules: {e}")
    
    def test_flask_app_exists(self):
        """Test if Flask app is properly initialized"""
        from app import app
        self.assertTrue(hasattr(app, 'run'))
        self.assertTrue(hasattr(app, 'route'))
    
    def test_config_exists(self):
        """Test if configuration exists"""
        from config import get_config
        config = get_config('development')
        self.assertIsNotNone(config)
        self.assertTrue(hasattr(config, 'UPLOAD_FOLDER'))
        self.assertTrue(hasattr(config, 'SECRET_KEY'))


class TestAppRoutes(unittest.TestCase):
    """Test app routes"""
    
    def setUp(self):
        """Set up test client"""
        from app import app
        self.app = app
        self.app.config['TESTING'] = True
        self.client = self.app.test_client()
    
    def test_health_endpoint(self):
        """Test /health endpoint"""
        response = self.client.get('/health')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'healthy')
    
    def test_index_endpoint(self):
        """Test / endpoint"""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
    
    def test_predict_no_image(self):
        """Test /predict with no image"""
        response = self.client.post('/predict')
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn('error', data)
    
    def test_404_error(self):
        """Test 404 error handling"""
        response = self.client.get('/nonexistent')
        self.assertEqual(response.status_code, 404)


class TestConfigModule(unittest.TestCase):
    """Test configuration module"""
    
    def test_development_config(self):
        """Test development configuration"""
        from config import DevelopmentConfig
        self.assertTrue(DevelopmentConfig.DEBUG)
        self.assertFalse(DevelopmentConfig.TESTING)
    
    def test_production_config(self):
        """Test production configuration"""
        from config import ProductionConfig
        self.assertFalse(ProductionConfig.DEBUG)
        self.assertFalse(ProductionConfig.TESTING)
    
    def test_get_config(self):
        """Test get_config function"""
        from config import get_config
        dev_config = get_config('development')
        prod_config = get_config('production')
        self.assertTrue(dev_config.DEBUG)
        self.assertFalse(prod_config.DEBUG)


class TestDetectors(unittest.TestCase):
    """Test detector modules"""
    
    def test_face_shape_detector_import(self):
        """Test FaceShapeDetector can be imported"""
        try:
            from detectors.face_shape_model import FaceShapeDetector
            self.assertIsNotNone(FaceShapeDetector)
        except ImportError as e:
            self.fail(f"Failed to import FaceShapeDetector: {e}")
    
    def test_gender_detector_import(self):
        """Test GenderDetector can be imported"""
        try:
            from detectors.gender_detection_model import GenderDetector
            self.assertIsNotNone(GenderDetector)
        except ImportError as e:
            self.fail(f"Failed to import GenderDetector: {e}")
    
    def test_hair_detector_import(self):
        """Test HairStyleDetector can be imported"""
        try:
            from detectors.hair_style_model import HairStyleDetector
            self.assertIsNotNone(HairStyleDetector)
        except ImportError as e:
            self.fail(f"Failed to import HairStyleDetector: {e}")
    
    def test_skin_detector_import(self):
        """Test SkinTypeDetector can be imported"""
        try:
            from detectors.skin_type_model import SkinTypeDetector
            self.assertIsNotNone(SkinTypeDetector)
        except ImportError as e:
            self.fail(f"Failed to import SkinTypeDetector: {e}")


class TestSuggestions(unittest.TestCase):
    """Test suggestions loading"""
    
    def test_suggestions_json_exists(self):
        """Test if suggestions.json exists"""
        suggestions_file = Path('grooming_suggestions/suggestions.json')
        self.assertTrue(suggestions_file.exists())
    
    def test_suggestions_json_valid(self):
        """Test if suggestions.json is valid"""
        suggestions_file = Path('grooming_suggestions/suggestions.json')
        with open(suggestions_file, 'r') as f:
            data = json.load(f)
            self.assertIn('face_shape', data)
            self.assertIn('gender', data)
            self.assertIn('hair_type', data)
            self.assertIn('skin_type', data)


class TestErrorHandling(unittest.TestCase):
    """Test error handling"""
    
    def setUp(self):
        """Set up test client"""
        from app import app
        self.app = app
        self.app.config['TESTING'] = True
        self.client = self.app.test_client()
    
    def test_invalid_content_type(self):
        """Test invalid content type"""
        response = self.client.post(
            '/predict',
            data={'invalid': 'data'},
            content_type='application/json'
        )
        self.assertIn(response.status_code, [400, 500])
    
    def test_empty_file(self):
        """Test empty file upload"""
        from werkzeug.datastructures import FileStorage
        from io import BytesIO
        
        empty_file = FileStorage(
            stream=BytesIO(b''),
            filename='empty.jpg',
            content_type='image/jpeg'
        )
        
        response = self.client.post(
            '/predict',
            data={'image': empty_file},
            content_type='multipart/form-data'
        )
        # Should handle gracefully
        self.assertIn(response.status_code, [200, 400, 500])


class TestPathsAndDirectories(unittest.TestCase):
    """Test project paths and directories"""
    
    def test_required_directories_exist(self):
        """Test if required directories exist"""
        required_dirs = [
            'templates',
            'static',
            'grooming_suggestions',
            'detectors'
        ]
        
        for directory in required_dirs:
            with self.subTest(directory=directory):
                self.assertTrue(
                    Path(directory).exists(),
                    f"Directory {directory} not found"
                )
    
    def test_required_files_exist(self):
        """Test if required files exist"""
        required_files = [
            'app.py',
            'predict.py',
            'config.py',
            'requirements.txt',
            'templates/index.html',
            'grooming_suggestions/suggestions.json'
        ]
        
        for file_path in required_files:
            with self.subTest(file=file_path):
                self.assertTrue(
                    Path(file_path).exists(),
                    f"File {file_path} not found"
                )


def run_tests():
    """Run all tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestAppInitialization))
    suite.addTests(loader.loadTestsFromTestCase(TestAppRoutes))
    suite.addTests(loader.loadTestsFromTestCase(TestConfigModule))
    suite.addTests(loader.loadTestsFromTestCase(TestDetectors))
    suite.addTests(loader.loadTestsFromTestCase(TestSuggestions))
    suite.addTests(loader.loadTestsFromTestCase(TestErrorHandling))
    suite.addTests(loader.loadTestsFromTestCase(TestPathsAndDirectories))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("="*70)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
