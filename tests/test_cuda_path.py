from whisper_fpmi.cuda_path import parse_gpu_list, pick_gpu


def test_pick_4060_when_2060_is_also_present():
    listing = "\n".join(
        [
            "0, NVIDIA GeForce RTX 4060 Ti",
            "1, NVIDIA GeForce RTX 2060",
        ]
    )
    assert pick_gpu(parse_gpu_list(listing)) == (0, "NVIDIA GeForce RTX 4060 Ti")


def test_pick_4060_even_if_it_is_not_the_first_card():
    listing = "\n".join(
        [
            "0, NVIDIA GeForce RTX 2060",
            "1, NVIDIA GeForce RTX 4060 Ti",
        ]
    )
    assert pick_gpu(parse_gpu_list(listing)) == (1, "NVIDIA GeForce RTX 4060 Ti")
