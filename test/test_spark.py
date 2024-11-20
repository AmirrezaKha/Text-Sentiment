import unittest
from external.scripts.spark.spark_main import DataIngestion
class TestSanitizeTableName(unittest.TestCase):
    def setUp(self):
        # Set up any common resources, if necessary
        self.obj = DataIngestion()

    def test_valid_table_name(self):
        # Test with a valid table name that doesn't require sanitization
        table_name = "valid_table_name"
        sanitized = self.obj.sanitize_table_name(table_name)
        self.assertEqual(sanitized, table_name, "The table name should remain unchanged.")

    def test_table_name_with_spaces(self):
        # Test table names with spaces
        table_name = "table with spaces"
        sanitized = self.obj.sanitize_table_name(table_name)
        self.assertEqual(sanitized, "table_with_spaces", "Spaces should be replaced with underscores.")

    def test_table_name_with_special_characters(self):
        # Test table names with special characters
        table_name = "table@name!123"
        sanitized = self.obj.sanitize_table_name(table_name)
        self.assertEqual(sanitized, "table_name_123", "Special characters should be removed or replaced.")

    def test_empty_table_name(self):
        # Test with an empty table name
        table_name = ""
        sanitized = self.obj.sanitize_table_name(table_name)
        self.assertEqual(sanitized, "default_table", "Empty names should default to a safe name.")

    def test_table_name_with_uppercase(self):
        # Test with uppercase letters
        table_name = "TableName"
        sanitized = self.obj.sanitize_table_name(table_name)
        self.assertEqual(sanitized, "tablename", "Uppercase letters should be converted to lowercase.")

if __name__ == "__main__":
    unittest.main()
