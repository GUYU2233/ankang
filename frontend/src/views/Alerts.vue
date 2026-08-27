<template>
  <el-card shadow="never">
    <template #header>
      <div class="card-header">
        <span>告警中心</span>
        <div>
          <el-radio-group v-model="filterLevel" size="small" @change="load">
            <el-radio-button label="">全部</el-radio-button>
            <el-radio-button label="red">红色</el-radio-button>
            <el-radio-button label="orange">橙色</el-radio-button>
            <el-radio-button label="yellow">黄色</el-radio-button>
          </el-radio-group>
          <el-button size="small" type="primary" @click="load" style="margin-left:8px;">刷新</el-button>
        </div>
      </div>
    </template>
    <el-table :data="alerts" size="small">
      <el-table-column prop="id" label="ID" width="60" />
      <el-table-column prop="level" label="级别" width="100">
        <template #default="s">
          <el-tag :type="levelType(s.row.level)" size="small">{{ levelText(s.row.level) }}</el-tag>
        </template>
      </el-table-column>
      <el-table-column prop="title" label="内容" />
      <el-table-column prop="event_type" label="类型" width="130" />
      <el-table-column prop="created_at" label="时间" width="190" />
      <el-table-column label="处置" width="130">
        <template #default="s">
          <el-button v-if="!s.row.handled" size="small" type="success" @click="ack(s.row)">确认处置</el-button>
          <el-tag v-else size="small" type="info">已处置</el-tag>
        </template>
      </el-table-column>
    </el-table>
  </el-card>
</template>

<script setup>
import { onMounted, ref } from 'vue'
import { ElMessage } from 'element-plus'
import api from '../api'

const alerts = ref([])
const filterLevel = ref('')

function levelType(level) { return { red: 'danger', orange: 'warning', yellow: 'warning', green: 'success' }[level] || 'info' }
function levelText(level) { return { red: '红色紧急', orange: '橙色高风险', yellow: '黄色关注', green: '绿色正常' }[level] || level }
async function load() {
  alerts.value = await api.alerts(filterLevel.value || null)
}
async function ack(row) {
  await api.ackAlert(row.id, true)
  ElMessage.success('已确认处置')
  load()
}
onMounted(load)
</script>

<style scoped>
.card-header { display: flex; justify-content: space-between; align-items: center; }
</style>