import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "extract_amoeba_task_dfgs",
    ROOT / "adapters" / "extract_amoeba_task_dfgs.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class ExtractTaskDfgTests(unittest.TestCase):
    def test_extracts_kernel_and_rebinds_outer_operands(self):
        source = """module {
  func.func @f(%arg0: memref<4xi32>) {
    taskflow.task @Task_0 will_reads(%arg0 : memref<4xi32>) {
    ^bb0(%arg1: memref<4xi32>):
      neura.kernel inputs(%arg1 : memref<4xi32>) attributes {accelerator = "neura"} {
      ^bb0(%arg2: memref<4xi32>):
        %0 = neura.counter -> !neura.data<index, i1>
        neura.yield
      }
      taskflow.yield
    }
    return
  }
}\n"""
        outputs = MODULE.extract_task_dfg_texts(source)
        self.assertEqual(set(outputs), {"Task_0"})
        self.assertIn("func.func @Task_0_dfg(%input0: memref<4xi32>)", outputs["Task_0"])
        self.assertIn("neura.kernel inputs(%input0 : memref<4xi32>)", outputs["Task_0"])
        self.assertNotIn("inputs(%arg1", outputs["Task_0"])

    def test_skips_non_neura_task(self):
        source = """module {
  func.func @f() {
    taskflow.task @CPU { taskflow.yield }
    taskflow.task @N {
      neura.kernel inputs(%x : i32) attributes {accelerator = "neura"} {
        neura.yield
      }
      taskflow.yield
    }
    return
  }
}\n"""
        self.assertEqual(set(MODULE.extract_task_dfg_texts(source)), {"N"})

    def test_extracts_iter_args_only_kernel(self):
        source = """module {
  func.func @f(%init: i32) {
    taskflow.task @Task_0 value_inputs(%init : i32) {
    ^bb0(%arg0: i32):
      %0 = neura.kernel iter_args_init(%arg0 : i32) attributes {accelerator = "neura"} {
      ^bb0(%arg1: i32):
        %1 = "neura.grant_once"() : () -> !neura.data<i32, i1>
        neura.yield
      } : i32
      taskflow.yield
    }
    return
  }
}\n"""
        output = MODULE.extract_task_dfg_texts(source)["Task_0"]
        self.assertIn("func.func @Task_0_dfg(%input0: i32)", output)
        self.assertIn("iter_args_init(%input0 : i32)", output)

    def test_extracts_inputs_and_iter_args_in_declared_order(self):
        source = """module {
  func.func @f(%x: i32, %init: f32) {
    taskflow.task @Task_0 {
      %0 = neura.kernel inputs(%x : i32) iter_args_init(%init : f32) attributes {accelerator = "neura"} {
      ^bb0(%arg0: i32, %arg1: f32):
        neura.yield
      } : f32
      taskflow.yield
    }
    return
  }
}\n"""
        output = MODULE.extract_task_dfg_texts(source)["Task_0"]
        self.assertIn("func.func @Task_0_dfg(%input0: i32, %input1: f32)", output)
        self.assertIn(
            "inputs(%input0 : i32) iter_args_init(%input1 : f32)",
            output,
        )

    def test_normalizes_zero_rank_store_indexed_to_generic_syntax(self):
        source = """module {
  func.func @f(%out: memref<i32>) {
    taskflow.task @Task_0 {
      neura.kernel inputs(%out : memref<i32>) attributes {accelerator = "neura"} {
      ^bb0(%arg0: memref<i32>):
        %0 = "neura.grant_once"() : () -> !neura.data<i32, i1>
        neura.store_indexed %0 to [ : ]  {rhs_value = "%input0"} : !neura.data<i32, i1>
        neura.yield
      }
      taskflow.yield
    }
    return
  }
}\n"""
        output = MODULE.extract_task_dfg_texts(source)["Task_0"]
        self.assertIn(
            '"neura.store_indexed"(%0) '
            '<{operandSegmentSizes = array<i32: 1, 0, 0>}> '
            '{rhs_value = "%input0"} : (!neura.data<i32, i1>) -> ()',
            output,
        )
        self.assertNotIn("to [ : ]", output)


if __name__ == "__main__":
    unittest.main()
