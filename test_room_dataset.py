import tempfile
import unittest
from pathlib import Path

import numpy as np

from room_dataset import (
    count_reflex_vertices,
    discretize_room_to_grid,
    generate_room_dataset,
    is_simple_polygon,
    load_room_dataset,
    polygon_to_edges,
    save_room_dataset,
    segments_intersect,
)


class TestRoomDataset(unittest.TestCase):
    def test_generated_rooms_are_simple(self):
        rooms = generate_room_dataset(n_rooms=100, min_vertices=4, max_vertices=10, seed=7)
        self.assertEqual(len(rooms), 100)

        for room in rooms:
            self.assertTrue(is_simple_polygon(room))

    def test_non_adjacent_walls_do_not_intersect(self):
        rooms = generate_room_dataset(n_rooms=20, min_vertices=6, max_vertices=9, seed=11)

        for room in rooms:
            edges = polygon_to_edges(room)
            n = edges.shape[0]

            for i in range(n):
                for j in range(i + 1, n):
                    if j == (i + 1) % n or i == (j + 1) % n:
                        continue

                    a, b = edges[i, 0], edges[i, 1]
                    c, d = edges[j, 0], edges[j, 1]
                    self.assertFalse(segments_intersect(a, b, c, d))

    def test_save_and_load_roundtrip(self):
        rooms = generate_room_dataset(n_rooms=5, min_vertices=4, max_vertices=6, seed=123)

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "rooms.json"
            save_room_dataset(p, rooms)
            loaded = load_room_dataset(p)

        self.assertEqual(len(rooms), len(loaded))
        for r0, r1 in zip(rooms, loaded):
            self.assertTrue(np.allclose(r0, r1))

    def test_hard_mode_rooms_are_non_convex_and_simple(self):
        rooms = generate_room_dataset(
            n_rooms=30,
            bounds=(-3.0, 3.0, -3.0, 3.0),
            seed=5,
            hard_mode=True,
            min_reflex_vertices=4,
            min_passage_width=1.0,
        )

        self.assertEqual(len(rooms), 30)
        for room in rooms:
            self.assertTrue(is_simple_polygon(room))
            self.assertGreaterEqual(count_reflex_vertices(room), 4)

    def test_hard_mode_invalid_width_raises(self):
        with self.assertRaises(ValueError):
            generate_room_dataset(
                n_rooms=1,
                bounds=(-0.4, 0.4, -0.4, 0.4),
                seed=1,
                hard_mode=True,
                min_passage_width=1.0,
            )

    def test_rectangle_intersection_mode_generates_simple_rooms(self):
        rooms = generate_room_dataset(
            n_rooms=15,
            bounds=(-15.0, 15.0, -15.0, 15.0),
            seed=9,
            rectangle_intersection_mode=True,
        )
        self.assertEqual(len(rooms), 15)
        for room in rooms:
            self.assertGreaterEqual(room.shape[0], 3)
            self.assertTrue(is_simple_polygon(room))
            self.assertGreater(abs(np.sum(room[:, 0] * np.roll(room[:, 1], -1) - room[:, 1] * np.roll(room[:, 0], -1))), 1e-3)

    def test_discretize_room_to_grid_respects_boundaries(self):
        room = np.array(
            [
                [-1.0, -0.5],
                [1.0, -0.5],
                [1.0, 0.5],
                [-1.0, 0.5],
            ],
            dtype=float,
        )

        north, south, east, west, centers = discretize_room_to_grid(room, dx=0.5, dy=0.5)

        # 2m by 1m rectangle with 0.5m cells should produce 4x2=8 cells.
        self.assertEqual(centers.shape[0], 8)
        self.assertEqual(north.shape[0], 8)
        self.assertEqual(south.shape[0], 8)
        self.assertEqual(east.shape[0], 8)
        self.assertEqual(west.shape[0], 8)

        self.assertTrue(np.all(centers[:, 0] >= -1.0 - 1e-9))
        self.assertTrue(np.all(centers[:, 0] <= 1.0 + 1e-9))
        self.assertTrue(np.all(centers[:, 1] >= -0.5 - 1e-9))
        self.assertTrue(np.all(centers[:, 1] <= 0.5 + 1e-9))


if __name__ == "__main__":
    unittest.main()
