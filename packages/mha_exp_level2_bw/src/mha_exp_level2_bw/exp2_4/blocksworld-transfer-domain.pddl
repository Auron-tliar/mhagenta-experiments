; Abstract transfer domain for experiment 2-4-BW.
;
; Arm position, arm contents, and table adjacency are intentionally absent.
; A grounded transfer records both supports so its successor predicts the exact
; On relation that the low-level atomic expansion must produce.

(define (domain abstract-blocksworld)
  (:requirements
    :strips
    :typing
    :fluents
    :negative-preconditions
    :equality
  )

  (:types
    block location
  )

  (:predicates
    (on ?block - block ?support - object)
    (clear ?support - object)
    (at-location ?item - object ?location - location)
  )

  (:functions
    (abstract-steps)
  )

  (:action transfer
    :parameters
      (?block - block
       ?source-support - object
       ?destination-support - object
       ?source - location
       ?destination - location)
    :precondition (and
      (on ?block ?source-support)
      (clear ?block)
      (at-location ?block ?source)
      (clear ?destination-support)
      (at-location ?destination-support ?destination)
      (not (= ?block ?destination-support))
      (not (= ?source ?destination))
    )
    :effect (and
      (not (on ?block ?source-support))
      (on ?block ?destination-support)
      (clear ?source-support)
      (not (clear ?destination-support))
      (not (at-location ?block ?source))
      (at-location ?block ?destination)
      (increase (abstract-steps) 1)
    )
  )
)
