import os

paths = [
    r"F:\NTUC-P-57_202506031303_202506031304.webm",
    r"F:\NTUC-P-57_202506031442_202506031445.webm",
    r"F:\NTUC-P-57_202506031631_202506031634.webm",
    r"F:\ruoxuan.webm",
    r"F:\NTUC-P-57_202506031838_202506031840.webm",
    r"F:\NTUC-P-57_202506041801_202506041803.webm",
    r"F:\NTUC-P-57_202506051744_202506051747.webm",
    r"F:\NTUC-P-57_202506061546_202506061553.webm",
    r"F:\NTUC-P-57_202506111624_202506111636.webm",
    r"C:\Users\Tairin Pairor\Downloads\NTUC-P-57_202506111749_202506111750.webm",
    r"C:\Users\Tairin Pairor\Downloads\NTUC-P-57_202506111759_202506111802.webm",
    r"C:\Users\Tairin Pairor\Downloads\NTUC-P-57_202506111811_202506111813.webm",
    r"C:\Users\Tairin Pairor\Downloads\NTUC-P-57_202506111814_202506111816.webm",
    r"C:\Users\Tairin Pairor\Downloads\NTUC-P-57_202506121531_202506121533.webm",
]

class VideoFrameEvaluator:
    def __init__(self):
        self.frame_ranges = {
            "NTUC-P-57_202506031303_202506031304": [(167, 239)],
            "NTUC-P-57_202506031442_202506031445": [(36, 209), (277, 659)],
            "NTUC-P-57_202506031631_202506031634": [(289, 543)],
            "ruoxuan": [(26, 220)],
            "NTUC-P-57_202506031838_202506031840": [],
            "NTUC-P-57_202506041801_202506041803": [(155, 283)],
            "NTUC-P-57_202506051744_202506051747": [(94, 424)],
            "NTUC-P-57_202506061546_202506061553": [(84, 1512)],
            "NTUC-P-57_202506111624_202506111636": [(28, 1091)],
            "NTUC-P-57_202506111749_202506111750": [(21, 333)],
            "NTUC-P-57_202506111759_202506111802": [(157, 517)],
            "NTUC-P-57_202506111811_202506111813": [(144, 339), (352, 425)],
            "NTUC-P-57_202506111814_202506111816": [(42, 171), (198, 251)],
            "NTUC-P-57_202506121531_202506121533": [(9, 512)],
        }

    def is_person_on_bed(self, video_path, frame_number):
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        ranges = self.frame_ranges.get(video_name, [])
        for start, end in ranges:
            if start <= frame_number < end:
                return video_path, video_name, 1
        return video_path, video_name, 0
    

# Example usage:
# labeler = VideoFrameEvaluator()
# print(labeler.is_person_on_bed(r"F:\NTUC-P-57_202506031442_202506031445.webm", 50))   # (path, name, 1)
# print(labeler.is_person_on_bed(r"F:\NTUC-P-57_202506031442_202506031445.webm", 210))  # (path, name, 0)
