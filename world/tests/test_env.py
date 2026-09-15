import numpy as np

from world.env import (
    ACTIONS,
    FIELDS,
    OUTCOMES,
    PASSAGE,
    SIZE,
    World,
    WorldConfig,
)


def test_every_cell_is_reachable_through_open_or_door_passages_and_lives_differ():
    a, b = World(seed=1), World(seed=2)
    for w in (a, b):
        for x in range(SIZE):
            for y in range(SIZE):
                assert w.shortest_path((0, 0), (x, y), doors_open=True) is not None
    assert a.state["edges"] != b.state["edges"]
    assert sum(v == "door_closed" for v in a.state["edges"].values()) == a.config.doors
    assert sum(v == "wall" for v in a.state["edges"].values()) == a.config.walls


def test_observation_exposes_eighteen_categorical_fields_and_only_visible_contents():
    w = World(seed=3)
    w.place_object(0, (1, 0))
    m = w.reset(agent=(0, 0))
    assert set(m.observation) == set(FIELDS)
    assert all(m.observation[k].shape == (n,) and m.observation[k].sum() == 1 for k, n in FIELDS.items())
    assert not m.observed["east_object"].any()  # neighbours are hidden before an inspect
    assert m.observation["here_object"].argmax() == 0
    d = w.act(0, ACTIONS.index("inspect"))
    if PASSAGE[m.observation["passage_east"].argmax()] != "wall":
        assert d.observed["east_object"].all() and d.observation["east_object"].argmax() == 1
    assert d.observation["outcome"].argmax() == OUTCOMES.index("inspected")


def test_moves_are_blocked_by_walls_and_closed_doors_and_interact_toggles_doors():
    w = World(seed=4)
    door = next(k for k, v in w.state["edges"].items() if v == "door_closed")
    (x, y), (x2, y2) = door
    direction = "east" if x2 > x else "south"
    w.reset(agent=(x, y))
    d = w.act(0, ACTIONS.index(direction))
    assert d.observation["outcome"].argmax() == OUTCOMES.index("blocked") and w.state["agent"] == (x, y)
    d = w.act(1, ACTIONS.index("interact"))
    assert d.observation["outcome"].argmax() == OUTCOMES.index("toggled")
    d = w.act(2, ACTIONS.index(direction))
    assert d.observation["outcome"].argmax() == OUTCOMES.index("moved") and w.state["agent"] == (x2, y2)
    w.set_door_rule("locked")
    d = w.act(3, ACTIONS.index("interact"))
    assert d.observation["outcome"].argmax() == OUTCOMES.index("none")


def test_objects_can_be_carried_and_dropped_and_feedback_names_the_executed_action():
    w = World(seed=5)
    w.place_object(2, (2, 2))
    m = w.reset(agent=(2, 2))
    assert m.observation["here_object"].argmax() == 3
    d = w.act(0, ACTIONS.index("interact"))
    assert d.observation["carrying"].argmax() == 3 and d.observation["here_object"].argmax() == 0
    assert d.feedback_for == 0 and d.executed == ACTIONS.index("interact") and d.reward_known
    d = w.act(1, ACTIONS.index("interact"))
    assert d.observation["carrying"].argmax() == 0 and w.state["objects"][2] == (2, 2)


def test_time_limit_truncates_with_a_final_observation_and_replay_is_deterministic():
    a, b = World(WorldConfig(horizon=5), seed=6), World(WorldConfig(horizon=5), seed=6)
    ma, mb = a.reset(), b.reset()
    for k, action in enumerate((0, 1, 4, 5, 2)):
        ma, mb = a.act(k, action), b.act(k, action)
        np.testing.assert_array_equal(ma.observation["x"], mb.observation["x"])
    assert ma.truncated and ma.final_observation is not None and not ma.terminated


def test_a_request_starts_a_new_episode_from_a_distant_cell_with_the_goal_set():
    from world.tasks import (
        GOAL_OBJECT,
        REQUEST_DISTANCE,
        Curriculum,
        manhattan,
    )

    w = World(seed=7)
    cur = Curriculum(w, seed=3)
    episode = cur.remember(8)
    m0 = w.reset(agent=episode.target_cell)
    assert m0.tick == 0 and int(m0.goal.argmax()) == 0
    m = cur.request(episode)
    assert m.tick == 0 and m.feedback_for is None and m.episode_id == m0.episode_id + 1
    assert int(m.goal.argmax()) == GOAL_OBJECT + episode.object_id
    assert manhattan(w.state["agent"], episode.target_cell) >= REQUEST_DISTANCE
    assert m.action_mask.all() and m.observed["x"].all()


def test_the_cue_decision_is_a_two_choice_episode_at_the_junction():
    from world.tasks import GOAL_CUE, ROOMS, Curriculum

    w = World(seed=7)
    cur = Curriculum(w, seed=3)
    episode = cur.cue(8)
    w.reset(agent=w.config.junction)
    m = cur.arm_cue(episode)
    assert m.tick == 0 and w.state["agent"] == w.config.junction and int(m.goal.argmax()) == GOAL_CUE
    assert set(np.flatnonzero(m.action_mask).tolist()) == {ACTIONS.index("north"), ACTIONS.index("west")}
    assert int(m.observation["word"].argmax()) == 0  # the cue is hidden: only memory can choose
    paid = episode.rewarded_room
    other = next(r for r in ROOMS if r != paid)
    direction = "north" if paid == (1, 0) else "west"
    m2 = w.act(0, ACTIONS.index(direction))
    assert m2.terminated and m2.reward == 1.0 and w.state["agent"] == paid
    w2 = World(seed=7)
    cur2 = Curriculum(w2, seed=3)
    e2 = cur2.cue(8)
    w2.reset(agent=w2.config.junction)
    cur2.arm_cue(e2)
    wrong = "west" if e2.rewarded_room == (1, 0) else "north"
    m3 = w2.act(0, ACTIONS.index(wrong))
    assert m3.terminated and m3.reward == 0.0 and w2.state["agent"] == other or m3.terminated and m3.reward == 0.0


def test_the_displacement_sensors_report_the_last_move():
    w = World(seed=7)
    m = w.reset(agent=(2, 2))
    assert int(m.observation["dx"].argmax()) == 1 and int(m.observation["dy"].argmax()) == 1
    for direction, (ex, ey) in (("east", (2, 1)), ("north", (1, 0)), ("west", (0, 1)), ("south", (1, 2))):
        w.reset(agent=(2, 2))
        m = w.act(0, ACTIONS.index(direction))
        moved = OUTCOMES[int(m.observation["outcome"].argmax())] == "moved"
        expect = (ex, ey) if moved else (1, 1)
        assert (int(m.observation["dx"].argmax()), int(m.observation["dy"].argmax())) == expect
    w.reset(agent=(2, 2))
    m = w.act(0, ACTIONS.index("inspect"))
    assert int(m.observation["dx"].argmax()) == 1 and int(m.observation["dy"].argmax()) == 1


def test_a_visible_move_lands_on_the_agents_empty_cell_and_is_in_view():
    from world.tasks import Curriculum

    w = World(seed=11)
    cur = Curriculum(w, seed=5)
    episode = cur.remember(16, move="visible")
    w.reset(agent=episode.target_cell)
    empty = next((x, y) for y in range(SIZE) for x in range(SIZE) if w.object_at((x, y)) is None)
    w.reset(agent=empty)
    cur.move_object(episode, visible=True)
    obs, observed = w.observation()
    assert w.state["objects"][episode.object_id] == empty
    assert int(obs["here_object"].argmax()) - 1 == episode.object_id
    assert "here_object" not in observed  # an absent flag map means the field is observed
    w.reset(agent=w.state["objects"][episode.object_id])
    import pytest

    with pytest.raises(ValueError):
        cur.move_object(episode, visible=True)
