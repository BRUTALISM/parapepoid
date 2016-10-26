(ns parapepoid.approach.core)

; An approach is a specific way of going about:
; 1. Generating color palettes
; 2. Mapping palette colors into neural network inputs

(defn approach
  "Creates an approach with the given palette generator function and input
  mapping transducer. The palette generator function should return a color and
  should have two arities – a 0-arity variant will be invoked for the first
  color being generated, while the 1-arity version will be invoked with the
  result of the previous iteration as a parameter. Think of the process as
  (iterate palette-fn (palette-fn)). The input mapping transducer should map a
  sequence of colors (representing one particular palette) into a sequence of
  floating point numbers."
  [palette-fn xinput]
  {::palette-generator palette-fn
   ::input-transducer xinput})

(defn palette
  "Creates a lazy, infinite color palette using the given approach."
  [approach]
  (let [palette-fn (::palette-generator approach)]
    (iterate palette-fn (palette-fn))))

(defn input-mapping
  "Returns the input transducer for the given approach."
  [approach]
  (::input-transducer approach))