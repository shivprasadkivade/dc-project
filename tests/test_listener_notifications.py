import unittest
from unittest.mock import patch
import platform

from src.holo.listen import show_toast, build_parser


class TestListenerNotifications(unittest.TestCase):

    @patch("src.holo.listen.subprocess.Popen")
    def test_show_toast_macos(self, mock_popen):
        with patch("platform.system", return_value="Darwin"):
            show_toast("AcousticTap - Copy (Cmd+C)", "Text Copied to Clipboard ✓", '"Hello World"', sound="Tink")
            mock_popen.assert_called_once()
            args = mock_popen.call_args[0][0]
            self.assertEqual(args[0], "osascript")
            self.assertEqual(args[1], "-e")
            self.assertIn("display notification", args[2])
            self.assertIn("Text Copied to Clipboard ✓", args[2])
            self.assertIn('sound name "Tink"', args[2])

    @patch("src.holo.listen.subprocess.Popen")
    def test_show_toast_non_macos_noop(self, mock_popen):
        with patch("platform.system", return_value="Linux"):
            show_toast("AcousticTap", "Subtitle", "Message")
            mock_popen.assert_not_called()

    def test_no_notify_arg_parsing(self):
        parser = build_parser()
        args = parser.parse_args(["--no-notify"])
        self.assertTrue(args.no_notify)

        args_default = parser.parse_args([])
        self.assertFalse(args_default.no_notify)


if __name__ == "__main__":
    unittest.main()
