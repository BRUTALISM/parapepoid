(ns parapepoid.util)

(defn select-values [m ks]
  (reduce #(conj %1 (%2 m)) [] ks))