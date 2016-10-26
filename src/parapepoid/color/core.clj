(ns parapepoid.color.core
  (:require [parapepoid.approach.core :as a]
            [thi.ng.color.core :as c]
            [thi.ng.math.core :as m]))

(defn as-rgb255-vec [color]
  ; TODO: Move to interop ns.
  (let [rgb (c/as-rgba color)]
    [(* (:r rgb) 255) (* (:g rgb) 255) (* (:b rgb) 255)]))

(defn generate-base-palette
  "Generates a base palette with count colors using the given approach."
  [approach count]
  (take count (a/palette approach)))

(defn next-color
  "Returns the next color in sequence for the given base colors and params."
  [colors params]
  (let [{:keys [hue saturation brightness]} params
        base (rand-nth colors)]
    (c/random-analog base hue saturation brightness)))

(defn infinite-palette
  "Creates a lazy infinite sequence of colors based off of the given base
  colors. The generation algorithm is configured using the params map."
  ([colors] (infinite-palette colors
                              {:hue 1.0 :saturation 1.0 :brightness 1.0}))
  ([colors params]
   (repeatedly #(next-color colors params))))
