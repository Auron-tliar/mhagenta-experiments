; Blocks World planning domain for experiment 2-3-BW.
;
; PDDL version: 2.1 (typed instantaneous actions with a numeric fluent).
;
; This domain mirrors the symbolic transition rules implemented by
; mha_env_blocksworld.BlocksWorldEnv. The environment is deterministic and
; fully observable, so generated problems use the closed-world assumption.
;
; Planner-to-environment action mapping
; -------------------------------------
; pickup    -> BlocksWorldEnv.Actions.PICK_UP    (0)
; putdown   -> BlocksWorldEnv.Actions.PUT_DOWN   (1)
; moveleft  -> BlocksWorldEnv.Actions.MOVE_LEFT  (2)
; moveright -> BlocksWorldEnv.Actions.MOVE_RIGHT (3)
;
; Problem-generation contract
; ---------------------------
; - Blocks and table locations are declared as objects of their respective
;   types. A location also acts as the immobile bottom support of its stack.
; - Every location ?l has (at-location ?l ?l).
; - (left-of ?left ?right) holds for consecutive table locations only.
; - Every block in a stack has exactly one (on ?block ?support) fact and one
;   (at-location ?block ?location) fact.
; - Each stack has exactly one clear top. For an empty stack, its location is
;   clear; otherwise, its top block is clear.
; - Exactly one (above ?location) fact is true.
; - Exactly one of these arm-state alternatives holds:
;     * (hand-empty), with no holding fact; or
;     * one (holding ?block) fact, with hand-empty false.
; - A held block has no on, clear, or at-location fact.
; - (elapsed-steps) is initialized to 0. A problem may use
;   (:metric minimize (elapsed-steps)) to prefer shorter plans.
;
; The historical thesis schema also named RightOf, NotAbove, IsBlock, and
; IsLoc. They are intentionally omitted here because the current runtime uses
; PDDL typing, represents rightward movement by reversing left-of arguments,
; and maintains a single above fact.

(define (domain blocksworld)
  (:requirements
    :strips
    :typing
    :fluents
  )

  (:types
    block location
  )

  (:predicates
    (hand-empty)
    (holding ?block - block)
    (on ?block - block ?support - object)
    (clear ?support - object)
    (above ?location - location)
    (at-location ?item - object ?location - location)
    (left-of ?left ?right - location)
  )

  (:functions
    (elapsed-steps)
  )

  (:action pickup
    :parameters
      (?block - block ?support - object ?location - location)
    :precondition (and
      (hand-empty)
      (on ?block ?support)
      (clear ?block)
      (at-location ?block ?location)
      (above ?location)
    )
    :effect (and
      (not (hand-empty))
      (not (on ?block ?support))
      (not (clear ?block))
      (not (at-location ?block ?location))
      (holding ?block)
      (clear ?support)
      (increase (elapsed-steps) 1)
    )
  )

  (:action putdown
    :parameters
      (?block - block ?support - object ?location - location)
    :precondition (and
      (holding ?block)
      (clear ?support)
      (at-location ?support ?location)
      (above ?location)
    )
    :effect (and
      (not (holding ?block))
      (not (clear ?support))
      (on ?block ?support)
      (clear ?block)
      (hand-empty)
      (at-location ?block ?location)
      (increase (elapsed-steps) 1)
    )
  )

  (:action moveleft
    :parameters (?from ?to - location)
    :precondition (and
      (above ?from)
      (left-of ?to ?from)
    )
    :effect (and
      (not (above ?from))
      (above ?to)
      (increase (elapsed-steps) 1)
    )
  )

  (:action moveright
    :parameters (?from ?to - location)
    :precondition (and
      (above ?from)
      (left-of ?from ?to)
    )
    :effect (and
      (not (above ?from))
      (above ?to)
      (increase (elapsed-steps) 1)
    )
  )
)
