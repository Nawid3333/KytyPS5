#include "graphics/host_gpu/renderer/labLoopGuard.h"

#include "common/logging/log.h"
#include "graphics/host_gpu/renderer/deviceLossDiag.h"

#include <algorithm>
#include <array>
#include <fmt/format.h>
#include <spirv-tools/libspirv.hpp>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <span>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace Libs::Graphics::Lab {
namespace {

constexpr uint32_t kSpirvMagic = 0x07230203u;

constexpr uint32_t kOpUndef = 1;
constexpr uint32_t kOpExtension = 10;
constexpr uint32_t kOpMemoryModel = 14;
constexpr uint32_t kOpCapability = 17;
constexpr uint32_t kOpTypeBool = 20;
constexpr uint32_t kOpTypeInt = 21;
constexpr uint32_t kOpTypePointer = 32;
constexpr uint32_t kOpConstant = 43;
constexpr uint32_t kOpFunction = 54;
constexpr uint32_t kOpFunctionEnd = 56;
constexpr uint32_t kOpVariable = 59;
constexpr uint32_t kOpLoad = 61;
constexpr uint32_t kOpStore = 62;
constexpr uint32_t kOpConvertUToPtr = 120;
constexpr uint32_t kOpIAdd = 128;
constexpr uint32_t kOpLogicalOr = 166;
constexpr uint32_t kOpLogicalAnd = 167;
constexpr uint32_t kOpLogicalNot = 168;
constexpr uint32_t kOpUGreaterThan = 172;
constexpr uint32_t kOpAtomicIAdd = 234;
constexpr uint32_t kOpAtomicOr = 241;
constexpr uint32_t kOpPhi = 245;
constexpr uint32_t kOpLoopMerge = 246;
constexpr uint32_t kOpSelectionMerge = 247;
constexpr uint32_t kOpLabel = 248;
constexpr uint32_t kOpBranch = 249;
constexpr uint32_t kOpBranchConditional = 250;
constexpr uint32_t kOpSwitch = 251;

constexpr uint32_t kTypeOpcodeFirst = 19;
constexpr uint32_t kTypeOpcodeLast = 39;
constexpr uint32_t kTypeOpcodeExtra0 = 322;
constexpr uint32_t kTypeOpcodeExtra1 = 4472;
constexpr uint32_t kTypeOpcodeExtra2 = 5341;
constexpr uint32_t kTypeOpcodeExtra3 = 5358;

constexpr uint32_t kCapabilityInt64 = 11;
constexpr uint32_t kCapabilityPhysicalStorageBuffer = 5347;
constexpr uint32_t kAddressingPhysicalStorageBuffer64 = 5348;
constexpr uint32_t kStorageClassFunction = 7;
constexpr uint32_t kStorageClassPhysicalStorageBuffer = 5349;

constexpr char kExtension[] = "SPV_KHR_physical_storage_buffer";

constexpr uint32_t kWordCountShift = 16;
constexpr uint32_t kOpcodeMask = 0xFFFFu;
constexpr uint32_t kHeaderWords = 5;
constexpr uint32_t kLowWordMask = 0xFFFFFFFFu;
constexpr uint32_t kLowWordBits = 32;
constexpr uint32_t kLastLoopBit = 31;
constexpr uint64_t kAddressCountOffset = 4;
constexpr uint32_t kWidth32 = 32;
constexpr uint32_t kWidth64 = 64;
constexpr uint32_t kSignednessUnsigned = 0;
constexpr uint32_t kSelectionControlNone = 0;

using Inst = std::vector<uint32_t>;

struct Block {
    uint32_t label = 0;
    std::vector<Inst> insts;
};

struct Function {
    std::vector<Inst> head;
    std::vector<Block> blocks;
    Inst end;
};

uint32_t Opcode(const Inst& inst) { return inst[0] & kOpcodeMask; }

Inst Make(uint32_t op, std::initializer_list<uint32_t> operands) {
    Inst inst;
    inst.reserve(operands.size() + 1);
    inst.push_back((static_cast<uint32_t>(operands.size() + 1) << kWordCountShift) | op);
    for (uint32_t operand : operands) inst.push_back(operand);
    return inst;
}

Inst MakeFromOperands(uint32_t op, const std::vector<uint32_t>& operands) {
    Inst inst;
    inst.reserve(operands.size() + 1);
    inst.push_back((static_cast<uint32_t>(operands.size() + 1) << kWordCountShift) | op);
    inst.insert(inst.end(), operands.begin(), operands.end());
    return inst;
}

template <typename Range>
bool TailEquals(const Inst& inst, size_t from, const Range& tail) {
    if (inst.size() != from + tail.size()) return false;
    size_t index = from;
    for (uint32_t word : tail) {
        if (inst[index] != word) return false;
        ++index;
    }
    return true;
}

std::vector<uint32_t> StringWords(const char* text) {
    std::vector<uint8_t> raw;
    for (const char* cursor = text; *cursor != '\0'; ++cursor) raw.push_back(static_cast<uint8_t>(*cursor));
    raw.push_back(0);
    while (raw.size() % 4 != 0) raw.push_back(0);
    std::vector<uint32_t> words(raw.size() / 4, 0);
    for (size_t index = 0; index < words.size(); ++index) {
        words[index] = static_cast<uint32_t>(raw[index * 4]) |
                       (static_cast<uint32_t>(raw[index * 4 + 1]) << 8) |
                       (static_cast<uint32_t>(raw[index * 4 + 2]) << 16) |
                       (static_cast<uint32_t>(raw[index * 4 + 3]) << 24);
    }
    return words;
}

bool IsTypeOpcode(uint32_t op) {
    if (op >= kTypeOpcodeFirst && op <= kTypeOpcodeLast) return true;
    return op == kTypeOpcodeExtra0 || op == kTypeOpcodeExtra1 || op == kTypeOpcodeExtra2 ||
           op == kTypeOpcodeExtra3;
}

bool HasKey(const std::unordered_map<uint32_t, uint32_t>& ids, uint32_t key) {
    return ids.find(key) != ids.end();
}

bool BlockHasLoopMerge(const Block& block) {
    for (const Inst& inst : block.insts)
        if (Opcode(inst) == kOpLoopMerge) return true;
    return false;
}

size_t BeforeMergeAndTerminator(const std::vector<Inst>& insts) {
    size_t index = insts.size() - 1;
    if (index > 0) {
        const uint32_t op = Opcode(insts[index - 1]);
        if (op == kOpLoopMerge || op == kOpSelectionMerge) index -= 1;
    }
    return index;
}

size_t AfterPhis(const std::vector<Inst>& insts) {
    size_t index = 1;
    while (index < insts.size() && Opcode(insts[index]) == kOpPhi) index += 1;
    return index;
}

std::vector<uint32_t> Successors(const Inst& term) {
    std::vector<uint32_t> out;
    const uint32_t op = Opcode(term);
    if (op == kOpBranch) {
        if (term.size() > 1) out.push_back(term[1]);
        return out;
    }
    if (op == kOpBranchConditional) {
        if (term.size() > 3) {
            out.push_back(term[2]);
            out.push_back(term[3]);
        }
        return out;
    }
    if (op == kOpSwitch) {
        if (term.size() > 2) {
            out.push_back(term[2]);
            for (size_t index = 4; index < term.size(); index += 2) out.push_back(term[index]);
        }
        return out;
    }
    return out;
}

std::unordered_map<uint32_t, size_t> BuildByLabel(const std::vector<Block>& blocks) {
    std::unordered_map<uint32_t, size_t> by_label;
    for (size_t index = 0; index < blocks.size(); ++index) by_label[blocks[index].label] = index;
    return by_label;
}

size_t FindFirstEqualBlock(const std::vector<Block>& blocks, size_t index) {
    const Block& target = blocks[index];
    for (size_t candidate = 0; candidate < blocks.size(); ++candidate) {
        if (blocks[candidate].label == target.label && blocks[candidate].insts == target.insts) return candidate;
    }
    return index;
}

uint32_t LoopBit(uint32_t g) { return g < kLastLoopBit ? g : kLastLoopBit; }

std::vector<uint32_t> Refuse(std::string* error, const std::string& text) {
    if (error != nullptr) *error = "bad module: " + text;
    return {};
}

struct Module {
    Inst header;
    uint32_t bound = 0;
    std::vector<Inst> globals;
    std::vector<Function> functions;
    std::vector<Inst> new_globals;
    std::unordered_map<uint32_t, uint32_t> type_ids;
    std::unordered_map<uint32_t, uint32_t> result_type;
    std::unordered_map<uint32_t, uint32_t> wide_ints;
    std::unordered_map<uint32_t, uint32_t> undefs;

    uint32_t NewId() {
        bound += 1;
        return bound - 1;
    }

    uint32_t TypeId(uint32_t op, std::initializer_list<uint32_t> operands) {
        for (int pass = 0; pass < 2; ++pass) {
            const std::vector<Inst>& list = pass == 0 ? globals : new_globals;
            for (const Inst& inst : list) {
                if (Opcode(inst) != op) continue;
                if (inst.size() < 2) continue;
                if (!TailEquals(inst, 2, operands)) continue;
                return inst[1];
            }
        }
        const uint32_t found = NewId();
        std::vector<uint32_t> words;
        words.reserve(operands.size() + 1);
        words.push_back(found);
        for (uint32_t operand : operands) words.push_back(operand);
        new_globals.push_back(MakeFromOperands(op, words));
        return found;
    }

    uint32_t Constant(uint32_t type, std::initializer_list<uint32_t> value) {
        const uint32_t result = NewId();
        std::vector<uint32_t> words;
        words.reserve(value.size() + 2);
        words.push_back(type);
        words.push_back(result);
        for (uint32_t word : value) words.push_back(word);
        new_globals.push_back(MakeFromOperands(kOpConstant, words));
        return result;
    }

    uint32_t Undef(uint32_t type) {
        auto it = undefs.find(type);
        if (it != undefs.end()) return it->second;
        const uint32_t id = NewId();
        undefs.emplace(type, id);
        new_globals.push_back(Make(kOpUndef, {type, id}));
        return id;
    }

    std::vector<uint32_t> Words() const {
        std::vector<uint32_t> out;
        out.push_back(header[0]);
        out.push_back(header[1]);
        out.push_back(header[2]);
        out.push_back(bound);
        out.push_back(header[4]);
        for (const Inst& inst : globals) out.insert(out.end(), inst.begin(), inst.end());
        for (const Inst& inst : new_globals) out.insert(out.end(), inst.begin(), inst.end());
        for (const Function& function : functions) {
            for (const Inst& inst : function.head) out.insert(out.end(), inst.begin(), inst.end());
            for (const Block& block : function.blocks)
                for (const Inst& inst : block.insts) out.insert(out.end(), inst.begin(), inst.end());
            out.insert(out.end(), function.end.begin(), function.end.end());
        }
        return out;
    }
};

void CollectResultType(Module& module, const Inst& inst) {
    if (IsTypeOpcode(Opcode(inst))) return;
    if (inst.size() < 3) return;
    if (!HasKey(module.type_ids, inst[1])) return;
    module.result_type[inst[2]] = inst[1];
}

bool ParseModule(std::span<const uint32_t> words, Module& module, std::string* why) {
    if (words.size() < kHeaderWords || words[0] != kSpirvMagic) {
        *why = "not a SPIR-V module";
        return false;
    }
    module.header.assign(words.begin(), words.begin() + kHeaderWords);
    module.bound = module.header[3];
    std::vector<Inst> insts;
    size_t cursor = kHeaderWords;
    while (cursor < words.size()) {
        const uint32_t count = words[cursor] >> kWordCountShift;
        if (count == 0 || cursor + count > words.size()) {
            *why = "bad instruction at word " + std::to_string(cursor);
            return false;
        }
        insts.push_back(Inst(words.begin() + cursor, words.begin() + cursor + count));
        cursor += count;
    }
    size_t first_function = insts.size();
    for (size_t index = 0; index < insts.size(); ++index) {
        if (Opcode(insts[index]) == kOpFunction) {
            first_function = index;
            break;
        }
    }
    module.globals.assign(insts.begin(), insts.begin() + first_function);
    std::vector<Inst> body(insts.begin() + first_function, insts.end());
    while (!body.empty()) {
        size_t end = body.size();
        for (size_t index = 0; index < body.size(); ++index) {
            if (Opcode(body[index]) == kOpFunctionEnd) {
                end = index;
                break;
            }
        }
        if (end == body.size()) {
            *why = "function without OpFunctionEnd";
            return false;
        }
        Function function;
        for (size_t index = 0; index < end; ++index) {
            const Inst& inst = body[index];
            if (Opcode(inst) == kOpLabel) {
                if (inst.size() < 2) {
                    *why = "OpLabel without a result id";
                    return false;
                }
                Block block;
                block.label = inst[1];
                block.insts.push_back(inst);
                function.blocks.push_back(std::move(block));
            } else if (!function.blocks.empty()) {
                function.blocks.back().insts.push_back(inst);
            } else {
                function.head.push_back(inst);
            }
        }
        function.end = body[end];
        module.functions.push_back(std::move(function));
        body.erase(body.begin(), body.begin() + static_cast<std::ptrdiff_t>(end + 1));
    }
    for (const Inst& inst : module.globals) {
        if (IsTypeOpcode(Opcode(inst)) && inst.size() > 1) module.type_ids.emplace(inst[1], inst[1]);
    }
    for (const Inst& inst : module.globals) CollectResultType(module, inst);
    for (const Function& function : module.functions)
        for (const Block& block : function.blocks)
            for (const Inst& inst : block.insts) CollectResultType(module, inst);
    for (const Inst& inst : module.globals) {
        if (Opcode(inst) != kOpTypeInt) continue;
        if (inst.size() < 3) {
            *why = "OpTypeInt with fewer than 3 words";
            return false;
        }
        if (inst[2] == kWidth64) module.wide_ints.emplace(inst[1], inst[1]);
    }
    return true;
}

} // namespace

std::vector<uint32_t> GuardLoops(std::span<const uint32_t> words, uint32_t cap, uint64_t address,
                                 std::vector<std::string>* report, std::string* error) {
    std::vector<std::string> lines;
    Module module;
    std::string why;
    if (!ParseModule(words, module, &why)) return Refuse(error, why);

    for (const Function& function : module.functions) {
        for (const Block& block : function.blocks) {
            if (block.insts.empty()) continue;
            const Inst& term = block.insts.back();
            if (Opcode(term) != kOpSwitch) continue;
            if (term.size() < 2) return Refuse(error, "OpSwitch without a selector");
            auto it = module.result_type.find(term[1]);
            if (it != module.result_type.end() && HasKey(module.wide_ints, it->second)) {
                return Refuse(error, "OpSwitch on a 64-bit selector");
            }
        }
    }

    const std::vector<uint32_t> extension_words = StringWords(kExtension);

    bool has_capability = false;
    for (const Inst& inst : module.globals) {
        if (Opcode(inst) == kOpCapability && inst.size() > 1 && inst[1] == kCapabilityPhysicalStorageBuffer) {
            has_capability = true;
        }
    }
    if (!has_capability) {
        size_t last = module.globals.size();
        for (size_t index = 0; index < module.globals.size(); ++index) {
            if (Opcode(module.globals[index]) == kOpCapability) last = index;
        }
        if (last == module.globals.size()) return Refuse(error, "no OpCapability in the module header");
        module.globals.insert(module.globals.begin() + static_cast<std::ptrdiff_t>(last + 1),
                              Make(kOpCapability, {kCapabilityPhysicalStorageBuffer}));
    }

    bool has_extension = false;
    for (const Inst& inst : module.globals) {
        if (Opcode(inst) == kOpExtension && TailEquals(inst, 1, extension_words)) has_extension = true;
    }
    if (!has_extension) {
        size_t last = module.globals.size();
        for (size_t index = 0; index < module.globals.size(); ++index) {
            const uint32_t op = Opcode(module.globals[index]);
            if (op == kOpCapability || op == kOpExtension) last = index;
        }
        if (last == module.globals.size()) {
            return Refuse(error, "no OpCapability or OpExtension in the module header");
        }
        module.globals.insert(module.globals.begin() + static_cast<std::ptrdiff_t>(last + 1),
                              MakeFromOperands(kOpExtension, extension_words));
    }

    size_t memory_model = module.globals.size();
    for (size_t index = 0; index < module.globals.size(); ++index) {
        if (Opcode(module.globals[index]) == kOpMemoryModel) {
            memory_model = index;
            break;
        }
    }
    if (memory_model == module.globals.size()) return Refuse(error, "no OpMemoryModel in the module header");
    if (module.globals[memory_model].size() < 2) {
        return Refuse(error, "OpMemoryModel without an addressing model");
    }
    module.globals[memory_model][1] = kAddressingPhysicalStorageBuffer64;

    bool has_int64_capability = false;
    for (const Inst& inst : module.globals) {
        if (Opcode(inst) == kOpCapability && inst.size() > 1 && inst[1] == kCapabilityInt64) {
            has_int64_capability = true;
        }
    }
    if (module.wide_ints.empty() && !has_int64_capability) {
        module.globals.insert(module.globals.begin(), Make(kOpCapability, {kCapabilityInt64}));
    }

    const uint32_t t_uint = module.TypeId(kOpTypeInt, {kWidth32, kSignednessUnsigned});
    const uint32_t t_ulong = module.TypeId(kOpTypeInt, {kWidth64, kSignednessUnsigned});
    const uint32_t t_bool = module.TypeId(kOpTypeBool, {});
    const uint32_t t_ptr_psb = module.NewId();
    module.new_globals.push_back(
        Make(kOpTypePointer, {t_ptr_psb, kStorageClassPhysicalStorageBuffer, t_uint}));
    const uint32_t t_ptr_fn = module.NewId();
    module.new_globals.push_back(Make(kOpTypePointer, {t_ptr_fn, kStorageClassFunction, t_uint}));
    const uint32_t c_zero = module.Constant(t_uint, {0});
    const uint32_t c_one = module.Constant(t_uint, {1});
    const uint32_t c_cap = module.Constant(t_uint, {cap});
    const uint32_t c_mask_address =
        module.Constant(t_ulong, {static_cast<uint32_t>(address & kLowWordMask),
                                  static_cast<uint32_t>(address >> kLowWordBits)});
    const uint64_t count_address = address + kAddressCountOffset;
    const uint32_t c_count_address =
        module.Constant(t_ulong, {static_cast<uint32_t>(count_address & kLowWordMask),
                                  static_cast<uint32_t>(count_address >> kLowWordBits)});

    struct LoopInfo {
        uint32_t g;
        size_t block;
        uint32_t merge;
        uint32_t cont;
    };

    size_t loop_index = 0;
    for (Function& function : module.functions) {
        std::vector<Block>& blocks = function.blocks;
        std::unordered_map<uint32_t, size_t> by_label = BuildByLabel(blocks);
        std::unordered_map<uint32_t, std::vector<uint32_t>> preds;
        for (const Block& block : blocks) preds[block.label] = std::vector<uint32_t>();
        for (const Block& block : blocks) {
            if (block.insts.empty()) return Refuse(error, "block without a terminator");
            for (uint32_t successor : Successors(block.insts.back())) preds[successor].push_back(block.label);
        }

        std::vector<LoopInfo> loops;
        for (size_t block_index = 0; block_index < blocks.size(); ++block_index) {
            const Block& block = blocks[block_index];
            if (block.insts.size() < 2) continue;
            if (Opcode(block.insts[block.insts.size() - 2]) != kOpLoopMerge) continue;
            const Inst& merge_inst = block.insts[block.insts.size() - 2];
            if (merge_inst.size() < 3) {
                return Refuse(error, "OpLoopMerge without a merge block or continue target");
            }
            const uint32_t merge = merge_inst[1];
            const uint32_t cont = merge_inst[2];
            const Inst& term = block.insts.back();
            const uint32_t g = static_cast<uint32_t>(loop_index);
            loop_index += 1;
            const char* reason = nullptr;
            bool cont_branches_to_header = false;
            auto pit = preds.find(block.label);
            if (pit != preds.end()) {
                for (uint32_t predecessor : pit->second) {
                    if (predecessor == cont) cont_branches_to_header = true;
                }
            }
            if (!cont_branches_to_header) {
                reason = "continue target does not branch to the header";
            } else {
                auto mit = by_label.find(merge);
                if (mit == by_label.end() || BlockHasLoopMerge(blocks[mit->second])) {
                    reason = "merge block missing or a loop header";
                } else if (Opcode(term) == kOpSwitch) {
                    reason = "header ends in OpSwitch";
                } else if (Opcode(term) == kOpBranchConditional) {
                    if (term.size() < 4) {
                        return Refuse(error, "OpBranchConditional with fewer than 4 words");
                    }
                    if (merge != term[2] && merge != term[3]) {
                        reason = "conditional header without an edge to the merge block";
                    }
                }
            }
            if (reason != nullptr) {
                lines.push_back("loop " + std::to_string(g) + " header %" + std::to_string(block.label) +
                                " merge %" + std::to_string(merge) + ": NOT guarded (" + reason + ")");
                continue;
            }
            loops.push_back({g, block_index, merge, cont});
        }
        if (loops.empty()) continue;

        std::unordered_map<uint32_t, uint32_t> counters;
        for (const LoopInfo& loop : loops) {
            const uint32_t counter = module.NewId();
            counters[loop.g] = counter;
            auto pit = preds.find(blocks[loop.block].label);
            if (pit == preds.end()) return Refuse(error, "loop header without predecessors");
            for (uint32_t predecessor : pit->second) {
                if (predecessor == loop.cont) continue;
                auto bit = by_label.find(predecessor);
                if (bit == by_label.end()) return Refuse(error, "predecessor block is missing");
                std::vector<Inst>& insts = blocks[bit->second].insts;
                insts.insert(insts.begin() + static_cast<std::ptrdiff_t>(BeforeMergeAndTerminator(insts)),
                             Make(kOpStore, {counter, c_zero}));
            }
        }

        for (const LoopInfo& loop : loops) {
            std::vector<Inst>& insts = blocks[loop.block].insts;
            const uint32_t v_old = module.NewId();
            const uint32_t v_new = module.NewId();
            const uint32_t over = module.NewId();
            const size_t at = AfterPhis(insts);
            std::vector<Inst> prologue;
            prologue.push_back(Make(kOpLoad, {t_uint, v_old, counters[loop.g]}));
            prologue.push_back(Make(kOpIAdd, {t_uint, v_new, v_old, c_one}));
            prologue.push_back(Make(kOpStore, {counters[loop.g], v_new}));
            prologue.push_back(Make(kOpUGreaterThan, {t_bool, over, v_new, c_cap}));
            insts.insert(insts.begin() + static_cast<std::ptrdiff_t>(at), prologue.begin(), prologue.end());
            const uint32_t term_op = Opcode(insts.back());
            if (term_op == kOpBranch) {
                if (insts.back().size() < 2) return Refuse(error, "OpBranch without a target label");
                const uint32_t target = insts.back()[1];
                if (target != loop.merge) {
                    insts.back() = Make(kOpBranchConditional, {over, loop.merge, target});
                    auto mit = by_label.find(loop.merge);
                    if (mit == by_label.end()) return Refuse(error, "merge block is missing");
                    std::vector<Inst>& merge_insts = blocks[mit->second].insts;
                    const size_t phis = AfterPhis(merge_insts);
                    for (size_t index = 1; index < phis; ++index) {
                        Inst& phi = merge_insts[index];
                        if (phi.size() < 2) return Refuse(error, "OpPhi without a result type");
                        const uint32_t undef = module.Undef(phi[1]);
                        phi.push_back(undef);
                        phi.push_back(blocks[loop.block].label);
                        phi[0] = (static_cast<uint32_t>(phi.size()) << kWordCountShift) | kOpPhi;
                    }
                }
            } else if (term_op == kOpBranchConditional) {
                if (insts.back().size() < 4) {
                    return Refuse(error, "OpBranchConditional with fewer than 4 words");
                }
                const uint32_t old_cond = insts.back()[1];
                const uint32_t yes_label = insts.back()[2];
                const uint32_t no_label = insts.back()[3];
                const std::vector<uint32_t> tail_operands(insts.back().begin() + 4, insts.back().end());
                const uint32_t cond = module.NewId();
                std::vector<Inst> extra;
                if (yes_label == loop.merge) {
                    extra.push_back(Make(kOpLogicalOr, {t_bool, cond, old_cond, over}));
                } else {
                    const uint32_t not_over = module.NewId();
                    extra.push_back(Make(kOpLogicalNot, {t_bool, not_over, over}));
                    extra.push_back(Make(kOpLogicalAnd, {t_bool, cond, old_cond, not_over}));
                }
                insts.insert(insts.end() - 2, extra.begin(), extra.end());
                std::vector<uint32_t> operands;
                operands.reserve(tail_operands.size() + 3);
                operands.push_back(cond);
                operands.push_back(yes_label);
                operands.push_back(no_label);
                operands.insert(operands.end(), tail_operands.begin(), tail_operands.end());
                insts.back() = MakeFromOperands(kOpBranchConditional, operands);
            }
            lines.push_back("loop " + std::to_string(loop.g) + " header %" +
                            std::to_string(blocks[loop.block].label) + " merge %" +
                            std::to_string(loop.merge) + " continue %" + std::to_string(loop.cont) +
                            ": guarded, bit " + std::to_string(LoopBit(loop.g)));
        }

        for (const LoopInfo& loop : loops) {
            auto mit = by_label.find(loop.merge);
            if (mit == by_label.end()) return Refuse(error, "merge block is missing");
            const size_t merge_at = FindFirstEqualBlock(blocks, mit->second);
            std::vector<Inst>& merge_insts = blocks[merge_at].insts;
            const size_t split = AfterPhis(merge_insts);
            std::vector<Inst> rest(merge_insts.begin() + static_cast<std::ptrdiff_t>(split),
                                   merge_insts.end());
            merge_insts.resize(split);
            if (rest.empty()) return Refuse(error, "merge block without a terminator");
            const std::vector<uint32_t> tail_successors = Successors(rest.back());
            const uint32_t tail_label = module.NewId();
            const uint32_t record_label = module.NewId();
            const uint32_t loaded = module.NewId();
            const uint32_t hit = module.NewId();
            const uint32_t p_mask = module.NewId();
            const uint32_t p_count = module.NewId();
            const uint32_t r_or = module.NewId();
            const uint32_t r_add = module.NewId();
            const uint32_t c_bit = module.Constant(t_uint, {1u << LoopBit(loop.g)});
            merge_insts.push_back(Make(kOpLoad, {t_uint, loaded, counters[loop.g]}));
            merge_insts.push_back(Make(kOpUGreaterThan, {t_bool, hit, loaded, c_cap}));
            merge_insts.push_back(Make(kOpSelectionMerge, {tail_label, kSelectionControlNone}));
            merge_insts.push_back(Make(kOpBranchConditional, {hit, record_label, tail_label}));
            Block record;
            record.label = record_label;
            record.insts.push_back(Make(kOpLabel, {record_label}));
            record.insts.push_back(Make(kOpConvertUToPtr, {t_ptr_psb, p_mask, c_mask_address}));
            record.insts.push_back(
                Make(kOpAtomicOr, {t_uint, r_or, p_mask, c_one, c_zero, c_bit}));
            record.insts.push_back(Make(kOpConvertUToPtr, {t_ptr_psb, p_count, c_count_address}));
            record.insts.push_back(
                Make(kOpAtomicIAdd, {t_uint, r_add, p_count, c_one, c_zero, c_one}));
            record.insts.push_back(Make(kOpBranch, {tail_label}));
            Block tail;
            tail.label = tail_label;
            tail.insts.push_back(Make(kOpLabel, {tail_label}));
            for (Inst& inst : rest) tail.insts.push_back(std::move(inst));
            blocks.insert(blocks.begin() + static_cast<std::ptrdiff_t>(merge_at) + 1, std::move(record));
            blocks.insert(blocks.begin() + static_cast<std::ptrdiff_t>(merge_at) + 2, std::move(tail));
            by_label = BuildByLabel(blocks);
            for (uint32_t successor : tail_successors) {
                auto sit = by_label.find(successor);
                if (sit == by_label.end()) return Refuse(error, "successor block is missing");
                std::vector<Inst>& successor_insts = blocks[sit->second].insts;
                const size_t phis = AfterPhis(successor_insts);
                for (size_t index = 1; index < phis; ++index) {
                    Inst& phi = successor_insts[index];
                    for (size_t slot = 4; slot < phi.size(); slot += 2) {
                        if (phi[slot] == loop.merge) phi[slot] = tail_label;
                    }
                }
            }
        }

        std::vector<Inst>& entry = blocks[0].insts;
        size_t at = 1;
        while (at < entry.size() && Opcode(entry[at]) == kOpVariable) at += 1;
        std::vector<Inst> variables;
        variables.reserve(loops.size());
        for (const LoopInfo& loop : loops) {
            variables.push_back(
                Make(kOpVariable, {t_ptr_fn, counters[loop.g], kStorageClassFunction, c_zero}));
        }
        entry.insert(entry.begin() + static_cast<std::ptrdiff_t>(at), variables.begin(), variables.end());
    }

    if (report != nullptr) report->insert(report->end(), lines.begin(), lines.end());
    return module.Words();
}

} // namespace Libs::Graphics::Lab

namespace Libs::Graphics::Lab {

namespace {

constexpr std::array<uint64_t, 4> GuardedShaders = {0xb37644325c6f10c6ull, 0x0dcde952329c979full,
                                                    0xcb0022ea5c2fc0cdull, 0x3e0139723a1fb10full};

} // namespace

vk::ShaderModule CompileGuardedSPV(std::span<const uint32_t> spirv, vk::Device device, uint64_t hash) {
	const uint64_t address = DeviceLossDiag::LoopGuardAddress();
	if (address == 0 || std::find(GuardedShaders.begin(), GuardedShaders.end(), hash) == GuardedShaders.end()) {
		return CompileSPV(spirv, device);
	}
	std::vector<std::string> report;
	std::string              error;
	const auto               guarded = GuardLoops(spirv, LoopGuardCap, address, &report, &error);
	if (guarded.empty()) {
		Log::WriteToConsoleAndLog(
		    fmt::format("[lab-guard] loop guard not applied to 0x{:016x}: {}\n", hash, error));
		return CompileSPV(spirv, device);
	}
	spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_3);
	std::string          messages;
	tools.SetMessageConsumer([&messages](spv_message_level_t, const char*, const spv_position_t&,
	                                     const char* message) { messages += message; });
	if (!tools.Validate(guarded.data(), guarded.size())) {
		Log::WriteToConsoleAndLog(fmt::format(
		    "[lab-guard] loop guard not applied to 0x{:016x}: validation failed: {}\n", hash, messages));
		return CompileSPV(spirv, device);
	}
	std::string lines;
	for (const auto& line: report) {
		lines += fmt::format("[lab-guard]   {}\n", line);
	}
	Log::WriteToConsoleAndLog(fmt::format("[lab-guard] loop guard on 0x{:016x} (cap {}, {} -> {} words):\n{}",
	                                      hash, LoopGuardCap, spirv.size(), guarded.size(), lines));
	return CompileSPV(guarded, device);
}

} // namespace Libs::Graphics::Lab
